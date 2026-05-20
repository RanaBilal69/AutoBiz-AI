"""
Inventory Agent - FIXED v6

"""
import math, json, re
from datetime import datetime, date
from typing import Optional, List
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from langchain_core.messages import AIMessage
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


def calculate_eoq(annual_demand, ordering_cost, holding_cost):
    if holding_cost <= 0:
        return 0
    return math.sqrt((2 * annual_demand * ordering_cost) / holding_cost)


# ── FIX 2: Multi-format date parser ──────────────────────────────────────────

_DATE_FORMATS = [
    "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y",
    "%m-%d-%Y", "%d-%m-%Y", "%Y/%m/%d",
    "%b %d, %Y", "%d %b %Y",
]

def _parse_date(val) -> Optional[date]:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date):
        return val
    raw = str(val).strip().split("T")[0].split(" ")[0]
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


def _deduplicate_items(items: List[dict]) -> List[dict]:
    seen_skus = set(); seen_names = set(); deduped = []
    for item in items:
        sku  = (item.get("sku")  or "").strip().upper()
        name = (item.get("name") or "").strip().lower()
        if sku and sku in seen_skus:
            continue
        if not sku and name and name in seen_names:
            continue
        if sku:
            seen_skus.add(sku)
        if name:
            seen_names.add(name)
        deduped.append(item)
    return deduped


# ── Read-only summary ─────────────────────────────────────────────────────────

def get_inventory_summary(
    db: Session = None,
    date_from: Optional[date] = None,
    date_to:   Optional[date] = None,
) -> str:
    """
    FIX 1: Reads ONLY from active source — never mixes CSV and DB.
    """
    source = get_active_source()

    if source == "csv":
        # FIX 1: CSV mode — never touch DB
        items = load_csv_data("inventory")
        if not items:
            return "No inventory data available."
        ctx      = "INVENTORY STATUS (CSV):\n"
        low_list = []
        for item in items:
            qty = int(item.get("quantity", 0))
            rp  = int(item.get("reorder_point", 0))
            st  = "LOW STOCK" if qty <= rp else "OK"
            if st == "LOW STOCK":
                low_list.append(item.get("name", ""))
            ctx += (
                f"  - {item.get('name')} (SKU:{item.get('sku','?')}, ID:{item.get('id','?')})\n"
                f"    Status:{st} | Stock:{qty} | Reorder at:{rp} | "
                f"Cost:${float(item.get('unit_cost', 0)):.2f} | "
                f"Supplier:{item.get('supplier', '?')}\n"
            )
        if low_list:
            ctx += f"\nLOW STOCK ITEMS: {', '.join(low_list)}\n"
        return ctx

    # DB mode — never touch CSV
    close = db is None
    if close:
        db = get_session()
    try:
        period_label = ""
        if date_from and date_to:
            period_label = f" (period {date_from} to {date_to})"
        elif date_from:
            period_label = f" (from {date_from})"
        elif date_to:
            period_label = f" (up to {date_to})"

        items = db.query(Inventory).all()
        if not items:
            return "No inventory data available."
        ctx      = f"INVENTORY STATUS{period_label}:\n"
        low_list = []
        for item in items:
            st = "OK"
            if item.quantity <= item.reorder_point:
                st = "LOW STOCK"
                low_list.append(item.name)
            eoq = calculate_eoq(
                item.quantity * 12, item.ordering_cost, item.holding_cost
            )
            ctx += (
                f"  - {item.name} (SKU:{item.sku}, ID:{item.id})\n"
                f"    Status:{st} | Stock:{item.quantity} | Reorder at:{item.reorder_point} | "
                f"Reorder qty:{item.reorder_quantity} | Cost:${item.unit_cost:.2f} | "
                f"Supplier:{item.supplier} (Lead:{item.supplier_lead_days}d) | EOQ:{eoq:.0f}\n"
            )
        if low_list:
            ctx += f"\nLOW STOCK ITEMS: {', '.join(low_list)}\n"
        return ctx
    finally:
        if close and db:
            db.close()


# ── Batch inventory detection ─────────────────────────────────────────────────

_BATCH_INV_RE = re.compile(
    r'(\d+\s+units?\s+of\s+\w[\w\s]*,.*?\d+\s+units?\s+of\s+\w[\w\s]*)'
    r'|'
    r'(\w[\w\s]+at\s+\$[\d.]+.*?,.*?\w[\w\s]+at\s+\$[\d.]+)'
    r'|'
    r'(\w[\w\s]+\d+\s+units?.*?,.*?\w[\w\s]+\d+\s+units?)'
    r'|'
    r'(restock|re-stock|update|set)\s+\w[\w\s]+to\s+\d+.*?,.*?\w[\w\s]+to\s+\d+',
    re.IGNORECASE | re.DOTALL
)

def is_batch_inventory_intent(user_input: str) -> bool:
    return bool(_BATCH_INV_RE.search(user_input))


# ── Write blocked message ─────────────────────────────────────────────────────

_CSV_WRITE_BLOCKED_MSG = (
    "Write operations are not available in CSV mode. "
    "CSV data is read-only. Please switch to database mode to add, "
    "update, or delete records."
)


# ── Batch inventory action ────────────────────────────────────────────────────

def execute_batch_inventory_action(
    user_message: str,
    db: Session,
    llm,
    history_ctx: str = "",
) -> str:
    # FIX 1: block in CSV mode
    if get_active_source() == "csv":
        return _CSV_WRITE_BLOCKED_MSG

    db_items   = db.query(Inventory).all()
    items_list = "\n".join(
        f"  ID={i.id} | Name={i.name} | SKU={i.sku} | Qty={i.quantity} | "
        f"ReorderPoint={i.reorder_point} | ReorderQty={i.reorder_quantity} | Cost={i.unit_cost}"
        for i in db_items
    )

    parse_prompt = f"""Parse this inventory request which contains MULTIPLE items.

CURRENT INVENTORY (DB):
{items_list}

{history_ctx}

REQUEST: {user_message}

Output a JSON ARRAY — one element per item:
{{
  "action": "create" | "update" | "restock",
  "item_id": <int or null>,
  "name": "<product name>",
  "sku": "<sku string>",
  "quantity": <int>,
  "reorder_point": <int or null>,
  "reorder_quantity": <int or null>,
  "unit_cost": <float or null>,
  "supplier": "<str or null>"
}}

CRITICAL:
- Output each item EXACTLY ONCE — no duplicate SKUs
- Existing SKU → action="update" or "restock" with item_id set
- New SKU → action="create" with item_id=null
- Do NOT output both restock and create for the same SKU
- Output ONLY the JSON array, no markdown"""

    raw      = invoke_with_fallback(llm, parse_prompt)
    raw_text = re.sub(r'```(?:json)?', '', raw.strip() if hasattr(raw, "strip") else str(raw)).strip()
    items_parsed = extract_json_array(raw_text) or (
        [extract_first_json(raw_text)] if extract_first_json(raw_text) else None
    )

    if not items_parsed:
        return "Could not parse batch inventory details."

    items_parsed = _deduplicate_items(items_parsed)
    results = []

    for item in items_parsed:
        action  = item.get("action", "create")
        item_id = item.get("item_id")
        int_id  = int(item_id) if item_id and str(item_id).isdigit() else None
        sku     = (item.get("sku")  or "").strip()
        name    = (item.get("name") or "").strip()

        inv = None
        if int_id:
            inv = db.query(Inventory).filter(Inventory.id == int_id).first()
        if not inv and sku:
            inv = db.query(Inventory).filter(Inventory.sku == sku).first()
        if not inv and name:
            inv = db.query(Inventory).filter(Inventory.name.ilike(f"%{name}%")).first()

        if action == "restock":
            if not inv:
                results.append(f"Item '{name or sku or '?'}' not found for restock.")
                continue
            old = inv.quantity
            inv.quantity = int(item.get("quantity", inv.reorder_quantity))
            if item.get("unit_cost") is not None:
                inv.unit_cost = float(item["unit_cost"])
            db.commit(); db.refresh(inv)
            st = "LOW STOCK" if inv.quantity <= inv.reorder_point else "OK"
            results.append(f"Restocked '{inv.name}' (SKU:{inv.sku}): {old} → {inv.quantity} units. Status: {st}.")

        elif action == "update":
            if not inv:
                results.append(f"Item '{name or sku or '?'}' not found for update.")
                continue
            changes = []
            for key, attr in [("quantity","quantity"),("reorder_point","reorder_point"),
                               ("reorder_quantity","reorder_quantity"),("unit_cost","unit_cost"),("supplier","supplier")]:
                val = item.get(key)
                if val is not None:
                    old = getattr(inv, attr)
                    setattr(inv, attr, val)
                    changes.append(f"{attr}: {old} → {val}")
            db.commit(); db.refresh(inv)
            results.append(f"Updated '{inv.name}' (SKU:{inv.sku}). Changes: {', '.join(changes) or 'none'}. Current qty: {inv.quantity}.")

        else:  # create
            if not sku:
                results.append(f"Cannot create '{name}': SKU is required.")
                continue
            if inv:
                if inv.name.lower() != name.lower() and name:
                    results.append(
                        f"SKU '{sku}' already belongs to '{inv.name}'. "
                        f"Cannot create '{name}' with the same SKU. Use a different SKU."
                    )
                else:
                    old = inv.quantity
                    inv.quantity = int(item.get("quantity", inv.quantity))
                    if item.get("unit_cost") is not None:
                        inv.unit_cost = float(item["unit_cost"])
                    db.commit(); db.refresh(inv)
                    st = "LOW STOCK" if inv.quantity <= inv.reorder_point else "OK"
                    results.append(f"Updated existing '{inv.name}' (SKU:{sku}): {old} → {inv.quantity} units. Status: {st}.")
                continue
            ni = Inventory(
                name=name or "New Item", sku=sku,
                quantity=int(item.get("quantity", 0)),
                reorder_point=int(item.get("reorder_point") or 10),
                reorder_quantity=int(item.get("reorder_quantity") or 50),
                unit_cost=float(item.get("unit_cost") or 0.0),
                holding_cost=1.0, ordering_cost=50.0,
                supplier=item.get("supplier"), supplier_lead_days=7,
            )
            db.add(ni); db.commit(); db.refresh(ni)
            results.append(f"Created '{ni.name}' (SKU:{ni.sku}, ID:{ni.id}) with {ni.quantity} units at ${ni.unit_cost}/unit.")

    return f"__ACTION__:inventory_write\n" + "\n".join(results)


# ── Single inventory action ───────────────────────────────────────────────────

def execute_inventory_action(
    user_message: str,
    db: Session,
    llm,
    history_ctx: str = "",
) -> str:
    # FIX 1: block in CSV mode
    if get_active_source() == "csv":
        return _CSV_WRITE_BLOCKED_MSG

    db_items   = db.query(Inventory).all()
    items_list = "\n".join(
        f"  ID={i.id} | Name={i.name} | SKU={i.sku} | Qty={i.quantity} | "
        f"ReorderPoint={i.reorder_point} | ReorderQty={i.reorder_quantity} | "
        f"Cost={i.unit_cost} | Supplier={i.supplier}"
        for i in db_items
    )

    parse_prompt = f"""Parse the inventory request and output ONLY valid JSON.

CURRENT INVENTORY (DB):
{items_list}

{history_ctx}

REQUEST: {user_message}

Output ONE of:
{{"action":"restock","item_id":<id>,"new_quantity":<int or null>,"reason":"<why>"}}
{{"action":"update","item_id":<id>,"fields":{{"quantity":<int>,"reorder_point":<int>,"unit_cost":<float>}},"reason":"<why>"}}
{{"action":"delete","item_id":<id>,"reason":"<why>"}}
{{"action":"create","name":"<str>","sku":"<str>","quantity":<int>,"reorder_point":<int>,"reorder_quantity":<int>,"unit_cost":<float>,"supplier":"<str or null>","reason":"<why>"}}
{{"action":"unknown","reason":"<why>"}}

Rules:
- Match item by name if no ID given.
- Output ONLY a single JSON object. No extra text."""

    raw    = invoke_with_fallback(llm, parse_prompt)
    parsed = extract_first_json(raw.strip() if hasattr(raw, "strip") else str(raw))
    if parsed is None:
        return "Could not parse action from request."

    action  = parsed.get("action", "unknown")
    item_id = parsed.get("item_id")
    int_id  = int(item_id) if item_id and str(item_id).isdigit() else None

    if action == "restock":
        item = db.query(Inventory).filter(Inventory.id == int_id).first()
        if not item:
            return f"Item ID {int_id} not found."
        old = item.quantity
        nq  = parsed.get("new_quantity") or max(item.reorder_quantity, item.reorder_point + item.reorder_quantity)
        item.quantity = int(nq)
        db.commit(); db.refresh(item)
        st = "LOW STOCK" if item.quantity <= item.reorder_point else "OK"
        return f"__ACTION__:inventory_write\nRestocked '{item.name}': {old} → {item.quantity} units. Status: {st}."

    elif action == "update":
        item = db.query(Inventory).filter(Inventory.id == int_id).first()
        if not item:
            return f"Item ID {int_id} not found."
        changes = []
        for field, value in (parsed.get("fields") or {}).items():
            if hasattr(item, field) and value is not None:
                old = getattr(item, field)
                setattr(item, field, value)
                changes.append(f"{field}: {old} → {value}")
        db.commit(); db.refresh(item)
        return f"__ACTION__:inventory_write\nUpdated '{item.name}'. Changes: {', '.join(changes)}. Current qty: {item.quantity}."

    elif action == "delete":
        item = db.query(Inventory).filter(Inventory.id == int_id).first()
        if not item:
            return f"Item ID {int_id} not found."
        name, sku = item.name, item.sku
        db.delete(item); db.commit()
        return f"__ACTION__:inventory_write\nDeleted '{name}' (SKU:{sku}) from inventory."

    elif action == "create":
        sku = (parsed.get("sku") or "").strip()
        if not sku:
            return "Cannot create item: SKU is required."
        if db.query(Inventory).filter(Inventory.sku == sku).first():
            return f"Item with SKU '{sku}' already exists."
        ni = Inventory(
            name=parsed.get("name", "New Item"), sku=sku,
            quantity=parsed.get("quantity", 0),
            reorder_point=parsed.get("reorder_point", 10),
            reorder_quantity=parsed.get("reorder_quantity", 50),
            unit_cost=parsed.get("unit_cost", 0.0),
            holding_cost=1.0, ordering_cost=50.0,
            supplier=parsed.get("supplier"), supplier_lead_days=7,
        )
        db.add(ni); db.commit(); db.refresh(ni)
        return f"__ACTION__:inventory_write\nCreated '{ni.name}' (SKU:{ni.sku}, ID:{ni.id}) with {ni.quantity} units at ${ni.unit_cost}/unit."

    return f"Action not recognized. Reason: {parsed.get('reason', 'unknown')}"


# ── Write intent detection ────────────────────────────────────────────────────

_WRITE_VERB_RE = re.compile(
    r'\b(add|create|insert|make|new|delete|remove|drop|erase|'
    r'update|edit|modify|change|set|increase|decrease|reduce|'
    r'adjust|restock|re-stock|refill|replenish)\b',
    re.IGNORECASE
)
_QUESTION_STARTER_RE = re.compile(
    r'^(show|what|which|how|tell|check|compare|list|give|find|'
    r'forecast|predict|analyze|analyse|summarize|report|its\s+end|end\s+of)\b',
    re.IGNORECASE
)

def is_write_intent(user_input: str) -> bool:
    stripped = user_input.strip()
    if _QUESTION_STARTER_RE.search(stripped):
        return False
    if stripped.endswith('?'):
        return False
    return bool(_WRITE_VERB_RE.search(stripped))


# ── Agent entrypoint ──────────────────────────────────────────────────────────

def inventory_agent(state: AgentState) -> AgentState:
    messages   = state["messages"]
    date_from  = state.get("date_from")
    date_to    = state.get("date_to")
    user_input = messages[-1].content
    source     = get_active_source()

    # ── WRITE path ────────────────────────────────────────────────────────────
    if is_write_intent(user_input):
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
        if is_batch_inventory_intent(user_input):
            result = execute_batch_inventory_action(user_input, db, llm)
        else:
            result = execute_inventory_action(user_input, db, llm)
        clean_result = result.replace("__ACTION__:inventory_write\n", "")
        db.close()
        return {
            "messages": messages + [AIMessage(content=clean_result)],
            "next":     "end",
            "response": result,
        }

    # ── READ path — FIX 1: only active source, never both ────────────────────
    llm     = get_llm()
    context = get_inventory_summary(date_from=date_from, date_to=date_to)

    period_note = ""
    if date_from and date_to:
        period_note = f" The user is asking about the period {date_from} to {date_to}."
    elif date_from:
        period_note = f" The user is asking about data from {date_from} onwards."
    elif date_to:
        period_note = f" The user is asking about data up to {date_to}."

    source_label = "CSV" if source == "csv" else "database"
    prompt = f"""You are an inventory management assistant. Data source: {source_label}.{period_note}
{context}
Question: {user_input}
Reply in 1-3 plain English sentences. Include key numbers. No *, **, #, - symbols."""
    r = invoke_with_fallback(llm, prompt)
    return {"messages": messages + [AIMessage(content=r)], "next": "end", "response": r}