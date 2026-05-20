"""
Workflow - FIXED v3

"""
import re
from langgraph.graph import StateGraph, END
from app.graph.state import AgentState
from app.agents.manager_agent import manager_agent
from app.agents.inventory_agent import (
    inventory_agent, execute_inventory_action,
    get_session as inv_get_session, get_inventory_summary
)
from app.agents.sales_agent import (
    sales_agent, execute_sales_action,
    get_session as sale_get_session
)
from app.core.llm import get_llm
from app.models.inventory import Inventory
from langchain_core.messages import AIMessage


# BUG C FIX: Bulk restock detection
_BULK_RESTOCK_RE = re.compile(
    r'\b(restock|re-stock|replenish|refill)\s+(all|every|each)\b'
    r'|'
    r'\ball\s+(low\s+stock|low-stock|out\s+of\s+stock)\b'
    r'|'
    r'\b(restock|re-stock)\s+all\b',
    re.IGNORECASE
)

def _is_bulk_restock(msg: str) -> bool:
    return bool(_BULK_RESTOCK_RE.search(msg))


def bulk_restock_action(state: AgentState) -> AgentState:
    """
    BUG C FIX: Restocks ALL low-stock items directly from DB.
    No LLM involved — finds every item where quantity <= reorder_point
    and sets quantity to reorder_quantity for each.
    """
    messages = state["messages"]
    db = inv_get_session()
    try:
        low_items = db.query(Inventory).filter(
            Inventory.quantity <= Inventory.reorder_point
        ).all()

        if not low_items:
            result = "All inventory items are already at or above reorder points. No restocking needed."
            db.close()
            return {
                "messages": messages + [AIMessage(content=result)],
                "next":     "end",
                "response": result,
            }

        lines = []
        for item in low_items:
            old_qty = item.quantity
            new_qty = max(item.reorder_quantity, item.reorder_point + item.reorder_quantity)
            item.quantity = new_qty
            db.commit()
            db.refresh(item)
            status = "OK" if item.quantity > item.reorder_point else "LOW STOCK"
            lines.append(
                f"Restocked '{item.name}' (SKU:{item.sku}): {old_qty} → {item.quantity} units. Status: {status}."
            )

        db.close()
        combined = "\n".join(lines)
        response = f"__ACTION__:inventory_write\n{combined}"
        return {
            "messages": messages + [AIMessage(content=combined)],
            "next":     "end",
            "response": response,
        }
    except Exception as e:
        db.close()
        err = f"Bulk restock failed: {str(e)}"
        return {"messages": messages + [AIMessage(content=err)], "next": "end", "response": err}


def multi_action_agent(state: AgentState) -> AgentState:
    """
    BUG A FIX: Passes targeted context to each step so LLM doesn't
    confuse shipment numbers with sale numbers.

    Step 1 — Inventory: tell LLM to extract ONLY the stock addition part.
    Step 2 — Sales: tell LLM explicitly what was already done, and to extract
              ONLY the sale transaction (quantity sold + sale amount), NOT
              the shipment quantity or shipment cost.
    """
    messages   = state["messages"]
    user_input = messages[-1].content
    llm        = get_llm()

    # ── Step 1: Inventory action ──────────────────────────────────────────────
    inv_db = inv_get_session()
    inv_result = execute_inventory_action(
        user_message=user_input,
        db=inv_db,
        llm=llm,
        history_ctx=(
            "STEP 1 of 2: Extract ONLY the inventory/stock addition part of this "
            "request (e.g. how many units were received/added and at what unit cost). "
            "Ignore any sale or sold information entirely."
        ),
    )
    inv_db.close()
    inv_clean = inv_result.replace("__ACTION__:inventory_write\n", "")

    # ── Step 2: Sales action — BUG A FIX: explicit sale-only context ─────────
    sale_db = sale_get_session()
    sale_result = execute_sales_action(
        user_message=user_input,
        db=sale_db,
        llm=llm,
        history_ctx=(
            "STEP 2 of 2: The inventory has already been updated in Step 1.\n"
            f"Step 1 result: {inv_clean}\n\n"
            "Now extract ONLY the SALE transaction from the original request.\n"
            "Rules:\n"
            "- Use the quantity that was SOLD (not the quantity received/shipped).\n"
            "- Use the SALE amount/price (not the purchase/unit cost from the shipment).\n"
            "- If the message says 'sold X units for $Y total', use X as quantity and Y as amount.\n"
            "- If the message says 'sold X units at $Z each', amount = X * Z.\n"
            "- Do NOT use the shipment quantity or shipment cost as the sale values.\n"
            "- action must be 'create' for a new sale record."
        ),
    )
    sale_db.close()
    sale_clean = sale_result.replace("__ACTION__:sales_write\n", "")

    combined_clean    = f"{inv_clean}\n{sale_clean}"
    combined_response = f"__ACTION__:inventory_write\n__ACTION__:sales_write\n{combined_clean}"

    return {
        "messages": messages + [AIMessage(content=combined_clean)],
        "next":     "end",
        "response": combined_response,
    }


def route_after_manager(state: AgentState) -> str:
    next_node  = state["next"]
    user_input = state["messages"][-1].content

    # BUG C FIX: bulk restock intercept before normal inventory routing
    if next_node in ("inventory_action", "inventory") and _is_bulk_restock(user_input):
        return "bulk_restock"

    if next_node in ("inventory_action", "inventory"):
        return "inventory"
    if next_node in ("sales_action", "sales"):
        return "sales"
    if next_node == "multi_action":
        return "multi_action"
    return "end"


# ── Build graph ───────────────────────────────────────────────────────────────

graph = StateGraph(AgentState)

graph.add_node("manager",      manager_agent)
graph.add_node("inventory",    inventory_agent)
graph.add_node("sales",        sales_agent)
graph.add_node("multi_action", multi_action_agent)
graph.add_node("bulk_restock", bulk_restock_action)   # BUG C FIX

graph.set_entry_point("manager")

graph.add_conditional_edges(
    "manager",
    route_after_manager,
    {
        "inventory":    "inventory",
        "sales":        "sales",
        "multi_action": "multi_action",
        "bulk_restock": "bulk_restock",
        "end":          END,
    }
)

graph.add_edge("inventory",    END)
graph.add_edge("sales",        END)
graph.add_edge("multi_action", END)
graph.add_edge("bulk_restock", END)

workflow = graph.compile()