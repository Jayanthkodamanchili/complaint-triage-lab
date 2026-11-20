# NEW CODE START
"""The Ops Assistant's 4 tools: small functions that look things up in evals/ops.db."""

import json
import sqlite3

OPS_DB_PATH = "evals/ops.db"
PRODUCTS = [
    "credit_reporting",
    "debt_collection",
    "checking_savings",
    "credit_card",
    "mortgage",
]


def ask_db(sql: str, params: tuple = ()) -> list[tuple]:
    """Open the notebook, run one question (SQL), close it (safe with Gradio threads)."""
    conn = sqlite3.connect(OPS_DB_PATH)
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return rows


# Part 1: the 4 tools (plain Python functions)
def get_top_issues(product: str | None = None, limit: int = 5) -> list[dict]:
    """The most common issues customers reported, optionally for one product."""
    sql = "SELECT customer_issue, COUNT(*) AS n FROM complaints"
    params: tuple = ()
    if product:
        sql += " WHERE product = ?"
        params = (product,)
    sql += " GROUP BY customer_issue ORDER BY n DESC LIMIT ?"
    rows = ask_db(sql, params + (limit,))
    return [{"issue": issue, "complaints": n} for issue, n in rows]


def count_by_urgency(product: str | None = None) -> dict:
    """How many complaints are high, medium or low urgency, optionally for one product."""
    sql = "SELECT urgency, COUNT(*) FROM complaints"
    params: tuple = ()
    if product:
        sql += " WHERE product = ?"
        params = (product,)
    sql += " GROUP BY urgency"
    return dict(ask_db(sql, params))


def compare_models(sort_by: str = "corrected_accuracy") -> list[dict]:
    """The leaderboard, sorted by accuracy, corrected accuracy, cost or speed."""
    columns = {
        "accuracy": '"accuracy_%" DESC',
        "corrected_accuracy": '"cleaned_acc_%" DESC',
        "cost": "usd_per_1m ASC",
        "speed": "median_sec ASC",
    }
    order = columns.get(sort_by, columns["corrected_accuracy"])
    rows = ask_db(
        'SELECT approach, "accuracy_%", "cleaned_acc_%", usd_per_1m, median_sec '
        f"FROM leaderboard ORDER BY {order}"
    )
    return [
        {
            "approach": approach,
            "accuracy_customer_labels_%": accuracy,
            "accuracy_corrected_labels_%": cleaned,
            "usd_per_1m_complaints": cost,
            "seconds_per_complaint": seconds,
        }
        for approach, accuracy, cleaned, cost, seconds in rows
    ]


def get_examples(
    product: str, urgency: str | None = None, limit: int = 3
) -> list[dict]:
    """A few example complaints (one-line summaries) for a product and urgency."""
    sql = "SELECT row_id, urgency, summary FROM complaints WHERE product = ?"
    params: tuple = (product,)
    if urgency:
        sql += " AND urgency = ?"
        params += (urgency,)
    sql += " ORDER BY row_id LIMIT ?"
    rows = ask_db(sql, params + (min(limit, 5),))
    return [{"id": r, "urgency": u, "summary": s} for r, u, s in rows]


# Part 2: the tool descriptions the LLM reads (Week 2 FlightAI format)
PRODUCT_FIELD = {
    "type": "string",
    "enum": PRODUCTS,
    "description": "One of the 5 products. Leave out for all products.",
}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_top_issues",
            "description": "Get the most common complaint issues, optionally for one product.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": PRODUCT_FIELD,
                    "limit": {
                        "type": "integer",
                        "description": "How many issues (1-10).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_by_urgency",
            "description": "Count complaints by urgency (high, medium, low), optionally for one product.",
            "parameters": {
                "type": "object",
                "properties": {"product": PRODUCT_FIELD},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_models",
            "description": "Compare the 6 approaches and the router on accuracy, cost and speed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sort_by": {
                        "type": "string",
                        "enum": ["accuracy", "corrected_accuracy", "cost", "speed"],
                        "description": "What to sort the leaderboard by.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_examples",
            "description": "Get up to 5 example complaint summaries for a product, optionally one urgency.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {"type": "string", "enum": PRODUCTS},
                    "urgency": {"type": "string", "enum": ["high", "medium", "low"]},
                    "limit": {
                        "type": "integer",
                        "description": "How many examples (1-5).",
                    },
                },
                "required": ["product"],
            },
        },
    },
]
TOOL_FUNCTIONS = {
    "get_top_issues": get_top_issues,
    "count_by_urgency": count_by_urgency,
    "compare_models": compare_models,
    "get_examples": get_examples,
}


# Part 3: run the tool the LLM asked for, and return the result as JSON text
def run_tool(name: str, arguments: str) -> str:
    try:
        result = TOOL_FUNCTIONS[name](**json.loads(arguments or "{}"))
    except Exception as error:  # noqa: BLE001 - tell the LLM what went wrong
        result = {"error": f"{type(error).__name__}: {error}"}
    return json.dumps(result)


# Part 4: self-test, no LLM involved ($0)
if __name__ == "__main__":
    print("Top issues (mortgage):", get_top_issues("mortgage", 3))
    print("Urgency (all):", count_by_urgency())
    print("Cheapest first:", [m["approach"] for m in compare_models("cost")][:3])
    print("Example:", get_examples("credit_card", "high", 1))
    print("Through run_tool:", run_tool("count_by_urgency", '{"product": "mortgage"}'))
# NEW CODE END
