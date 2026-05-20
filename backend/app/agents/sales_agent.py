"""
Sales Agent - FIXED v5

"""
import json, re
from datetime import datetime, date
from typing import Optional, List
from sqlalchemy import create_engine, func, text
from sqlalchemy.orm import sessionmaker, Session
from langchain_core.messages import AIMessage
from app.models.sales import Sales
from app.models.inventory import Inventory
from app.graph.state import AgentState
from app.core.llm import get_llm, invoke_with_fallback
from app.core.db_config import (
    get_current_db_url, get_active_source, load_csv_data
)


def get_session() -> Session:
    url    = get_current_db_url()
    engine = create_engine(url, pool_pre_ping=True)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)()


# ── FIX 2: Multi-format date parser ──────────────────────────────────────────

# All formats tried in order. M/D/YYYY must come before D/M/YYYY because
# CSV screenshot shows "4/12/2026" which is M/D/YYYY (April 12).
_DATE_FORMATS = [
    "%Y-%m-%d",   # 2026-04-12  (DB / ISO)
    "%m/%d/%Y",   # 4/12/2026   (CSV screenshot format)
    "%d/%m/%Y",   # 12/4/2026   (EU format)
    "%m-%d-%Y",   # 04-12-2026
    "%d-%m-%Y",   # 12-04-2026
    "%Y/%m/%d",   # 2026/04/12
    "%b %d, %Y",  # Apr 12, 2026
    "%d %b %Y",   # 12 Apr 2026
]

def _parse_date(val) -> Optional[date]:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date):
        return val
    raw = str(val).strip()
    # Strip time portion if present (e.g. "2026-04-12T00:00:00" or "2026-04-12 00:00:00")
    raw = raw.split("T")[0].split(" ")[0]
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


# ── JSON extractors ───────────────────────────────────────────────────────────

def extract_first_json(text: str) -> Optional[dict]:
    start = text.find('{')
    if start == -1:
        return None
    depth = 0; in_str = False; escape = False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False; continue
        if ch == '\\' and in_str:
            escape = True; continue
        if ch == '"':
            in_str = not in_str; continue
        if in_str:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def extract_json_array(text: str) -> Optional[List[dict]]:
    start = text.find('[')
    if start == -1:
        return None
    depth = 0; in_str = False; escape = False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False; continue
        if ch == '\\' and in_str:
            escape = True; continue
        if ch == '"':
            in_str = not in_str; continue
        if in_str:
            continue
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                try:
                    result = json.loads(text[start:i + 1])
                    if isinstance(result, list):
                        return result
                except json.JSONDecodeError:
                    return None
    return None


# ── Batch sale detection ──────────────────────────────────────────────────────

_BATCH_SALE_RE = re.compile(
    r'(\d+\s+\w[\w\s]*at\s+\$[\d.]+.*?,.*?\d+\s+\w[\w\s]*at\s+\$[\d.]+)'
    r'|'
    r'(sold\s+\d+.*?,\s*\d+.*?,\s*and\s+\d+)'
    r'|'
    r'(\d+\s+\w[\w\s]*,\s*\d+\s+\w[\w\s]*,\s*(and\s+)?\d+\s+\w[\w\s]*)',
    re.IGNORECASE
)

def is_batch_sale_intent(user_input: str) -> bool:
    return bool(_BATCH_SALE_RE.search(user_input))


# ── Read-only summary ─────────────────────────────────────────────────────────

def get_sales_summary(
    db: Session = None,
    date_from: Optional[date] = None,
    date_to:   Optional[date] = None,
) -> str:
    """
    FIX 1: Reads ONLY from active source.
    FIX 2: Uses multi-format _parse_date() so CSV date strings don't return None.
    """
    source = get_active_source()
    close  = db is None

    if source == "csv":
        # FIX 1: CSV mode — never touch DB
        all_sales = load_csv_data("sales")
    else:
        # DB mode — never touch CSV
        if close:
            db = get_session()
        try:
            raw = db.query(Sales).all()
            all_sales = [
                {
                    "product":   s.product,
                    "sku":       s.sku,
                    "quantity":  s.quantity,
                    "amount":    s.amount,
                    "category":  s.category,
                    "sale_date": str(s.sale_date) if s.sale_date else None,
                }
                for s in raw
            ]
        finally:
            if close and db:
                db.close()

    if not all_sales:
        return "No sales data available."

    # FIX 2: date filtering now works for all CSV date formats
    if date_from or date_to:
        filtered = []
        unparseable = 0
        for s in all_sales:
            sd = _parse_date(s.get("sale_date"))
            if sd is None:
                unparseable += 1
                continue   # genuinely unparseable — skip
            if date_from and sd < date_from:
                continue
            if date_to   and sd > date_to:
                continue
            filtered.append(s)
        sales = filtered
    else:
        sales = all_sales

    if not sales:
        period = ""
        if date_from and date_to:
            period = f" between {date_from} and {date_to}"
        elif date_from:
            period = f" from {date_from} onwards"
        elif date_to:
            period = f" up to {date_to}"
        return f"No sales data found{period}."

    total_rev   = sum(float(s.get("amount")   or 0) for s in sales)
    total_units = sum(int(s.get("quantity") or 0) for s in sales)
    pr = {}; pu = {}; cr = {}
    for s in sales:
        p   = s.get("product", "Unknown")
        amt = float(s.get("amount")   or 0)
        qty = int(s.get("quantity") or 0)
        pr[p] = pr.get(p, 0) + amt
        pu[p] = pu.get(p, 0) + qty
        cat = s.get("category") or ""
        if cat:
            cr[cat] = cr.get(cat, 0) + amt

    period_label = ""
    if date_from and date_to:
        period_label = f" ({date_from} to {date_to})"
    elif date_from:
        period_label = f" (from {date_from})"
    elif date_to:
        period_label = f" (up to {date_to})"

    ctx = (
        f"SALES SUMMARY{period_label}:\n"
        f"Total Revenue: ${total_rev:,.2f}\n"
        f"Units: {total_units}\n"
        f"Transactions: {len(sales)}\n\nBY PRODUCT:\n"
    )
    for p, r in sorted(pr.items(), key=lambda x: x[1], reverse=True):
        ctx += f"  {p}: ${r:,.2f} ({pu[p]} units)\n"
    ctx += "\nBY CATEGORY:\n"
    for c, r in sorted(cr.items(), key=lambda x: x[1], reverse=True):
        ctx += f"  {c}: ${r:,.2f}\n"
    ctx += "\nRECENT:\n"
    recent = sorted(
        sales, key=lambda x: str(x.get("sale_date") or ""), reverse=True
    )[:5]
    for s in recent:
        ctx += f"  {s.get('product')}: {s.get('quantity')} units @ ${float(s.get('amount') or 0):,.2f}\n"
    return ctx


def generate_sql(user_input: str, llm) -> str:
    try:
        sql_prompt = f"""Generate a PostgreSQL SELECT query.
Table: sales | Columns: id,product,sku,quantity,amount,category,sale_date
Return ONLY the SQL, no markdown. Question: {user_input}"""
        r   = llm.invoke(sql_prompt)
        sql = r.content.strip().replace('```sql', '').replace('```', '').strip()
        return sql if sql.upper().startswith('SELECT') else None
    except Exception:
        return None


def execute_sql(sql: str, db: Session) -> str:
    try:
        result = db.execute(text(sql))
        rows   = result.fetchall()
        cols   = list(result.keys())
        if not rows:
            return ""
        out = " | ".join(cols) + "\n" + "-" * 40 + "\n"
        for row in rows[:20]:
            out += " | ".join(str(v) for v in row) + "\n"
        return out
    except Exception:
        return ""


# ── Write actions — DB ONLY, BLOCKED in CSV mode ──────────────────────────────

_CSV_WRITE_BLOCKED_MSG = (
    "Write operations are not available in CSV mode. "
    "CSV data is read-only. Please switch to database mode to add, "
    "update, or delete records."
)


def execute_batch_sales_action(
    user_message: str,
    db: Session,
    llm,
    history_ctx: str = "",
) -> str:
    # FIX 1: block writes in CSV mode
    if get_active_source() == "csv":
        return _CSV_WRITE_BLOCKED_MSG

    inv_items = db.query(Inventory).all()
    inv_list  = "\n".join(
        f"  ID={i.id}|Name={i.name}|SKU={i.sku}|Qty={i.quantity}"
        for i in inv_items
    )

    parse_prompt = f"""Parse this sales request which contains MULTIPLE products being sold.

INVENTORY (from DB):
{inv_list}

{history_ctx}

REQUEST: {user_message}

Output a JSON ARRAY where each element is one sale:
[
  {{"product":"Monitor","sku":null,"quantity":15,"amount":4800.00,"category":"Electronics"}},
  {{"product":"Keyboard","sku":null,"quantity":8,"amount":440.00,"category":"Electronics"}}
]

Rules:
- quantity = units sold for that product
- amount = total revenue (quantity * unit price)
- Include ALL products mentioned — do not skip any
- Output ONLY the JSON array, no extra text, no markdown"""

    raw      = invoke_with_fallback(llm, parse_prompt)
    raw_text = re.sub(r'```(?:json)?', '', raw.strip() if hasattr(raw, "strip") else str(raw)).strip()
    sales_list = extract_json_array(raw_text) or ([extract_first_json(raw_text)] if extract_first_json(raw_text) else None)

    if not sales_list:
        return "Could not parse batch sale details."

    results = []
    for item in sales_list:
        product  = (item.get("product") or "").strip()
        quantity = int(item.get("quantity") or 0)
        amount   = float(item.get("amount") or 0.0)
        if not product or quantity <= 0 or amount <= 0:
            results.append(f"Skipped invalid entry: {item}")
            continue
        sale = Sales(
            product=product, sku=item.get("sku") or None,
            quantity=quantity, amount=amount,
            category=item.get("category") or None,
            sale_date=datetime.utcnow(),
        )
        db.add(sale); db.commit(); db.refresh(sale)
        inv_result = _deduct_inventory(db, inv_items, product, item.get("sku"), quantity)
        results.append(
            f"Sale recorded: {quantity} units of '{product}' for ${amount:,.2f} (ID:{sale.id}). {inv_result}"
        )

    return f"__ACTION__:sales_write\n" + "\n".join(results)


def execute_sales_action(
    user_message: str,
    db: Session,
    llm,
    history_ctx: str = "",
) -> str:
    # FIX 1: block writes in CSV mode
    if get_active_source() == "csv":
        return _CSV_WRITE_BLOCKED_MSG

    raw_sales = db.query(Sales).order_by(Sales.id.desc()).limit(30).all()
    sales_list = "\n".join(
        f"  ID={s.id}|Product={s.product}|SKU={s.sku or ''}|"
        f"Qty={s.quantity}|Amount={s.amount}|Date={s.sale_date}"
        for s in raw_sales
    )
    inv_items = db.query(Inventory).all()
    inv_list  = "\n".join(
        f"  ID={i.id}|Name={i.name}|SKU={i.sku}|Qty={i.quantity}"
        for i in inv_items
    )

    parse_prompt = f"""Parse this sales management request and output ONLY valid JSON.

CURRENT SALES (last 30, from DB):
{sales_list}

INVENTORY (from DB):
{inv_list}

{history_ctx}

REQUEST: {user_message}

Output ONE of:
{{"action":"create","product":"<name>","sku":"<sku or null>","quantity":<int>,"amount":<float>,"category":"<category or null>"}}
{{"action":"update","sale_id":<int>,"fields":{{"product":"<str>","quantity":<int>,"amount":<float>,"category":"<str>"}}}}
{{"action":"delete","sale_id":<int>,"reason":"<why>"}}
{{"action":"delete_by_product","product":"<name>","reason":"<why>"}}

Rules:
- quantity = units SOLD (not received/shipped)
- amount = TOTAL sale revenue
- Output ONLY a single JSON object. No extra text."""

    raw    = invoke_with_fallback(llm, parse_prompt)
    parsed = extract_first_json(raw.strip() if hasattr(raw, "strip") else str(raw))
    if parsed is None:
        return "Could not parse sale details."

    action = parsed.get("action", "create")

    if action == "delete":
        sale_id = parsed.get("sale_id")
        if not sale_id:
            return "Delete failed: no sale ID specified."
        sale = db.query(Sales).filter(Sales.id == sale_id).first()
        if not sale:
            return f"Sale ID {sale_id} not found."
        product = sale.product
        db.delete(sale); db.commit()
        return f"__ACTION__:sales_write\nDeleted sale record ID:{sale_id} for '{product}'."

    if action == "delete_by_product":
        product_name = parsed.get("product", "").strip().lower()
        if not product_name:
            return "Delete failed: no product name specified."
        rows = db.query(Sales).filter(func.lower(Sales.product).contains(product_name)).all()
        if not rows:
            return f"No sales records found for product '{product_name}'."
        count = len(rows)
        for row in rows:
            db.delete(row)
        db.commit()
        return f"__ACTION__:sales_write\nDeleted {count} sale record(s) for '{product_name}'."

    if action == "update":
        sale_id = parsed.get("sale_id")
        fields  = parsed.get("fields", {})
        if not sale_id:
            return "Update failed: no sale ID specified."
        sale = db.query(Sales).filter(Sales.id == sale_id).first()
        if not sale:
            return f"Sale ID {sale_id} not found."
        changes = []
        for field, value in fields.items():
            if field in {"product","sku","quantity","amount","category"} and value is not None:
                old = getattr(sale, field)
                setattr(sale, field, value)
                changes.append(f"{field}: {old} → {value}")
        db.commit(); db.refresh(sale)
        return f"__ACTION__:sales_write\nUpdated sale ID:{sale_id} ({sale.product}). Changes: {', '.join(changes)}."

    # CREATE
    product  = parsed.get("product", "").strip()
    quantity = int(parsed.get("quantity", 0))
    amount   = float(parsed.get("amount", 0.0))
    if not product or quantity <= 0 or amount <= 0:
        return "Invalid sale details: need product name, quantity > 0, and amount > 0."

    inv_items = db.query(Inventory).all()
    sale = Sales(
        product=product, sku=parsed.get("sku") or None,
        quantity=quantity, amount=amount,
        category=parsed.get("category") or None,
        sale_date=datetime.utcnow(),
    )
    db.add(sale); db.commit(); db.refresh(sale)
    inv_result = _deduct_inventory(db, inv_items, product, parsed.get("sku"), quantity)
    return (
        f"__ACTION__:sales_write\n"
        f"Sale recorded: {quantity} units of '{product}' for ${amount:,.2f} (ID:{sale.id}). {inv_result}"
    )


def _deduct_inventory(db, inv_items, product_name, sku, qty):
    inv_item = None
    if sku:
        inv_item = db.query(Inventory).filter(Inventory.sku == sku).first()
    if not inv_item:
        for i in inv_items:
            if product_name.lower() in i.name.lower() or i.name.lower() in product_name.lower():
                inv_item = db.query(Inventory).filter(Inventory.id == i.id).first()
                break
    if not inv_item:
        return "Note: no matching inventory item found to deduct stock."
    old = inv_item.quantity
    inv_item.quantity = max(0, inv_item.quantity - qty)
    db.commit(); db.refresh(inv_item)
    st = "LOW STOCK" if inv_item.quantity <= inv_item.reorder_point else "OK"
    return f"Inventory '{inv_item.name}' updated: {old} → {inv_item.quantity}. Status: {st}."


# ── Write intent ──────────────────────────────────────────────────────────────

SALES_WRITE_KEYWORDS = [
    "record sale", "add sale", "log sale", "new sale",
    "delete sale", "remove sale", "delete sales", "remove sales",
    "update sale", "edit sale", "modify sale", "change sale",
    "mark as sold", "sold ", "record a sale",
    "delete sales record", "remove sales record",
    "delete sale record", "remove sale record",
    "delete the sale", "remove the sale",
    "from sales", "in sales",
    "log all", "log them", "we sold", "all today",
    "log all of them", "record all", "record them",
]

_DELETE_FROM_SALES_RE = re.compile(
    r'\b(delete|remove)\b.{0,60}\b(from|in)\s+sales\b', re.IGNORECASE
)

def is_sales_write_intent(user_input: str) -> bool:
    if _DELETE_FROM_SALES_RE.search(user_input):
        return True
    return any(kw in user_input.lower() for kw in SALES_WRITE_KEYWORDS)


# ── Agent entrypoint ──────────────────────────────────────────────────────────

def sales_agent(state: AgentState) -> AgentState:
    messages   = state["messages"]
    date_from: Optional[date] = state.get("date_from")
    date_to:   Optional[date] = state.get("date_to")
    user_input = messages[-1].content
    source     = get_active_source()

    # ── WRITE path ────────────────────────────────────────────────────────────
    if is_sales_write_intent(user_input):
        # FIX 1: CSV mode — block writes immediately, no DB session opened
        if source == "csv":
            reply = _CSV_WRITE_BLOCKED_MSG
            return {
                "messages": messages + [AIMessage(content=reply)],
                "next":     "end",
                "response": reply,
            }

        db  = get_session()
        llm = get_llm()
        if is_batch_sale_intent(user_input):
            result = execute_batch_sales_action(user_input, db, llm)
        else:
            result = execute_sales_action(user_input, db, llm)
        clean_result = result.replace("__ACTION__:sales_write\n", "")
        db.close()
        return {
            "messages": messages + [AIMessage(content=clean_result)],
            "next":     "end",
            "response": result,
        }

    # ── READ path ─────────────────────────────────────────────────────────────
    llm = get_llm()

    # FIX 1: CSV mode — ONLY use CSV summary, never open DB or run SQL
    if source == "csv":
        context = get_sales_summary(date_from=date_from, date_to=date_to)
        data    = f"SALES DATA (CSV):\n{context}"
    else:
        # DB mode — use SQL + DB summary, never touch CSV
        db          = get_session()
        sql_results = ""
        try:
            sql = generate_sql(user_input, llm)
            if sql:
                sql_results = execute_sql(sql, db)
        except Exception:
            pass
        context = get_sales_summary(db, date_from=date_from, date_to=date_to)
        db.close()
        data = (
            f"QUERY RESULTS:\n{sql_results}\n\nSALES DATA:\n{context}"
            if sql_results else f"SALES DATA:\n{context}"
        )

    period_note = ""
    if date_from and date_to:
        period_note = f" The user is asking about the period {date_from} to {date_to}. Only use data from that range."
    elif date_from:
        period_note = f" The user is asking about data from {date_from} onwards. Only use data from that range."
    elif date_to:
        period_note = f" The user is asking about data up to {date_to}. Only use data from that range."

    prompt = f"""You are a sales analysis assistant.{period_note}
{data}
Question: {user_input}
Reply in 1-3 plain English sentences. Include key numbers. No *, **, #, - symbols."""
    r = invoke_with_fallback(llm, prompt)
    return {"messages": messages + [AIMessage(content=r)], "next": "end", "response": r}