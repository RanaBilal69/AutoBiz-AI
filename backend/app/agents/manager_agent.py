"""
Manager Agent - FIXED v4

"""
import re
from datetime import date, datetime, timedelta
from calendar import monthrange
from typing import Optional, Tuple
from langchain_core.messages import AIMessage
from app.graph.state import AgentState

# ── Keyword lists ─────────────────────────────────────────────────────────────

SALES_KW = [
    "sale", "revenue", "selling", "sold", "earning", "income", "profit", "order",
    "customer", "trend", "category", "transaction", "best sell", "top product",
    "monthly", "forecast sales", "predict sales", "average order",
]
INVENTORY_KW = [
    "stock", "inventory", "reorder", "supplier", "warehouse", "eoq", "lead time",
    "low stock", "out of stock", "reorder point", "unit cost", "sku",
    "items", "goods", "restocking", "replenish", "refill",
]

ANALYSIS_QUESTION_PATTERNS = [
    r'\bshow\s+me\b',
    r'\bwhat\s+(are|is|should|were|was|will|would)\b',
    r'\bwhich\s+(products?|items?|categories?)\b',
    r'\btell\s+me\b',
    r'\bhow\s+much\b',
    r'\bhow\s+many\b',
    r'\bcompare\b',
    r'\bforecast\b',
    r'\bpredict\b',
    r'\banalyze\b',
    r'\banalyse\b',
    r'\bsummary\b',
    r'\bsummarize\b',
    r'\breport\b',
    r'\bperforming\b',
    r'\bperformance\b',
    r'\btrend\b',
    r'\bgrowth\b',
    r'\bits\s+end\s+of\b',
    r'\bend\s+of\s+(the\s+)?(month|year|week|quarter)\b',
    r'\bcheck\s+(the\s+)?(inventory|stock|sales)\b',
]
_ANALYSIS_RE = re.compile('|'.join(ANALYSIS_QUESTION_PATTERNS), re.IGNORECASE)

ACTION_SALE = [
    "record sale", "add sale", "log sale", "new sale",
    "delete sale", "remove sale", "delete sales", "remove sales",
    "update sale", "edit sale", "modify sale", "change sale",
    "mark as sold", "record a sale", "sold ",
    "delete sales record", "remove sales record",
    "delete the sale", "remove the sale",
    "delete sale record", "remove sale record",
    "from sales", "in sales", "update sales",
    # BUG D FIX: natural batch-sale language
    "log all", "log them", "we sold", "all today",
    "log all of them", "record all", "record them",
]

ACTION_INV = [
    "restock", "refill", "replenish",
    "add stock", "increase stock", "update item", "edit item",
    "change quantity", "set quantity",
    "delete item", "remove item",
    "adjust", "add item", "create item",
    "new item", "update stock", "edit stock",
    "decrease stock", "reduce stock",
    "update quantity", "edit quantity", "modify quantity", "set stock",
    "create product", "add product", "new product",
    "delete product", "remove product",
    "update product", "edit product",
    "re-stock", "restock all", "re-stock all",
]

MULTI_INV_KW = [
    "add stock", "increase stock", "add more", "restock", "update stock",
    "update quantity", "add units", "more units", "more stock",
    "received", "shipment", "stock arrived", "delivery arrived", "just arrived",
    "re-stock",
]
MULTI_SALE_KW = [
    "record a sale", "record sale", "add sale", "log sale",
    "sold", "sell", "sale of", "immediately sold", "then sold",
]

_DELETE_FROM_SALES_RE = re.compile(
    r'\b(delete|remove)\b.{0,60}\b(from|in)\s+sales\b', re.IGNORECASE
)
_ADD_UNITS_RE = re.compile(
    r'\badd\s+\d+\s*(?:units?|pieces?|pcs?|items?)\b', re.IGNORECASE
)
_RECEIVED_AND_SOLD_RE = re.compile(
    r'\b(received|shipment|stock\s+arrived|delivery)\b.{0,300}\b(sold|sell|sale)\b',
    re.IGNORECASE | re.DOTALL
)

# BUG D FIX: "15 Monitors at $320 each, 8 Keyboards at $55 each" pattern
_BATCH_SALE_ROUTE_RE = re.compile(
    r'\d+\s+\w[\w\s]*at\s+\$[\d,.]+\s+each.*?,.*?\d+\s+\w[\w\s]*at\s+\$[\d,.]+',
    re.IGNORECASE | re.DOTALL
)

NON_ENGLISH_PATTERN = re.compile(
    r'[\u0600-\u06FF\u0750-\u077F'
    r'\u4e00-\u9fff'
    r'\u0900-\u097F'
    r'\u0400-\u04FF'
    r'\u3040-\u30ff'
    r'\uAC00-\uD7A3]'
)

MONTH_MAP = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6,
    "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9,
    "sep": 9, "sept": 9, "october": 10, "oct": 10,
    "november": 11, "nov": 11, "december": 12, "dec": 12,
}


def _is_non_english(text: str) -> bool:
    return bool(NON_ENGLISH_PATTERN.search(text))


def _today() -> date:
    return datetime.utcnow().date()


def _is_analysis_question(msg: str) -> bool:
    return bool(_ANALYSIS_RE.search(msg))


def _parse_date_range(message: str) -> Tuple[Optional[date], Optional[date]]:
    msg   = message.lower().strip()
    today = _today()

    m = re.search(r'\blast\s+(\d+)\s+days?\b', msg)
    if m:
        n = int(m.group(1))
        return today - timedelta(days=n - 1), today

    m = re.search(r'\blast\s+(\d+)\s+months?\b', msg)
    if m:
        n = int(m.group(1))
        y, mo = today.year, today.month
        mo -= n
        while mo <= 0:
            mo += 12; y -= 1
        return date(y, mo, 1), today

    if re.search(r'\bthis\s+month\b', msg):
        first = today.replace(day=1)
        return first, today.replace(day=monthrange(today.year, today.month)[1])

    if re.search(r'\blast\s+month\b', msg):
        first_this = today.replace(day=1)
        last_month = first_this - timedelta(days=1)
        return last_month.replace(day=1), last_month.replace(
            day=monthrange(last_month.year, last_month.month)[1]
        )

    if re.search(r'\bthis\s+week\b', msg):
        start = today - timedelta(days=today.weekday())
        return start, start + timedelta(days=6)

    if re.search(r'\blast\s+week\b', msg):
        start_this = today - timedelta(days=today.weekday())
        end = start_this - timedelta(days=1)
        return end - timedelta(days=6), end

    if re.search(r'\bthis\s+year\b', msg):
        return date(today.year, 1, 1), date(today.year, 12, 31)

    if re.search(r'\blast\s+year\b', msg):
        y = today.year - 1
        return date(y, 1, 1), date(y, 12, 31)

    if re.search(r'\bend\s+of\s+(the\s+)?year\b', msg):
        return date(today.year, 1, 1), date(today.year, 12, 31)

    if re.search(r'\bend\s+of\s+(the\s+)?month\b', msg):
        first = today.replace(day=1)
        return first, today.replace(day=monthrange(today.year, today.month)[1])

    iso = re.search(
        r'(\d{4}-\d{2}-\d{2})\s*(?:to|-|through|until)\s*(\d{4}-\d{2}-\d{2})', msg
    )
    if iso:
        try:
            return date.fromisoformat(iso.group(1)), date.fromisoformat(iso.group(2))
        except ValueError:
            pass

    iso_single = re.search(r'\b(\d{4}-\d{2}-\d{2})\b', msg)
    if iso_single:
        try:
            d = date.fromisoformat(iso_single.group(1))
            return d, d
        except ValueError:
            pass

    year_only = re.search(r'\b(in|for|of|during)\s+(20\d{2}|19\d{2})\b', msg)
    if year_only:
        y = int(year_only.group(2))
        return date(y, 1, 1), date(y, 12, 31)

    month_pattern = r'\b(' + '|'.join(MONTH_MAP.keys()) + r')\b'
    m = re.search(month_pattern, msg)
    if m:
        month_num = MONTH_MAP[m.group(1)]
        year_m    = re.search(r'\b(20\d{2}|19\d{2})\b', msg)
        year      = int(year_m.group(1)) if year_m else today.year
        first     = date(year, month_num, 1)
        return first, date(year, month_num, monthrange(year, month_num)[1])

    return None, None


def _is_multi_action(msg: str) -> bool:
    lower = msg.lower()
    if _RECEIVED_AND_SOLD_RE.search(msg):
        return True
    has_inv  = any(kw in lower for kw in MULTI_INV_KW) or bool(_ADD_UNITS_RE.search(msg))
    has_sale = any(kw in lower for kw in MULTI_SALE_KW)
    return has_inv and has_sale


def classify(message: str) -> str:
    msg = message.lower().strip()

    # Analysis questions → read nodes only
    if _is_analysis_question(message):
        inv_s  = sum(1 for k in INVENTORY_KW if k in msg)
        sale_s = sum(1 for k in SALES_KW     if k in msg)
        return "inventory" if (inv_s > 0 and inv_s >= sale_s) else "sales"

    # BUG D FIX: batch sale pattern → sales_action
    if _BATCH_SALE_ROUTE_RE.search(message):
        return "sales_action"

    if _DELETE_FROM_SALES_RE.search(message):
        return "sales_action"

    if any(p in msg for p in ACTION_SALE):
        return "sales_action"

    if any(p in msg for p in ACTION_INV):
        return "inventory_action"

    if _ADD_UNITS_RE.search(message) and "sale" not in msg:
        return "inventory_action"

    inv_s  = sum(1 for k in INVENTORY_KW if k in msg)
    sale_s = sum(1 for k in SALES_KW     if k in msg)

    if inv_s > 0 and inv_s >= sale_s:
        return "inventory"
    if sale_s > 0:
        return "sales"
    return "end"


def manager_agent(state: AgentState) -> AgentState:
    messages   = state["messages"]
    user_input = messages[-1].content

    if _is_non_english(user_input):
        reply = "This system handles English language queries only."
        return {
            "messages":  messages + [AIMessage(content=reply)],
            "next":      "end",
            "response":  reply,
            "date_from": None,
            "date_to":   None,
        }

    # Analysis questions skip multi_action check
    if not _is_analysis_question(user_input) and _is_multi_action(user_input):
        date_from, date_to = _parse_date_range(user_input)
        return {
            "messages":  messages,
            "next":      "multi_action",
            "response":  "",
            "date_from": date_from,
            "date_to":   date_to,
        }

    decision = classify(user_input)

    if decision == "end":
        reply = "This system handles business queries about sales and inventory only."
        return {
            "messages":  messages + [AIMessage(content=reply)],
            "next":      "end",
            "response":  reply,
            "date_from": None,
            "date_to":   None,
        }

    date_from, date_to = _parse_date_range(user_input)

    return {
        "messages":  messages,
        "next":      decision,
        "response":  "",
        "date_from": date_from,
        "date_to":   date_to,
    }