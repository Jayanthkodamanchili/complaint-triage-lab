# NEW CODE START
"""Pack 5 real complaints + the answers our LLMs already gave into one small file."""

import json

import pandas as pd

from complaint_triage.db import connect, get_cached, make_key
from complaint_triage.prompts import build_messages

OUTPUT_PATH = "evals/showcase.json"

# Part 1: the 5 complaints we show the public (one per product, picked by hand)
SHOWCASE = {
    2278244: "Identity theft items on my credit report",
    2849484: "Collector chasing a stranger's apartment debt",
    422461: "Bank froze my checking and savings money",
    542566: "Card dispute denied for charms never delivered",
    2555654: "Mortgage balance grew during Covid forbearance",
}

# Part 2: the 4 LLMs (name shown in the app -> model name in the receipt book)
LLM_MODELS = {
    "Claude Haiku 5.5": "claude-haiku-5-5",
    "GPT-6 Luna": "gpt-6-luna",
    "Gemini Flash-Lite": "gemini-flash-lite-latest",
    "llama3.2 (local)": "llama3.2",
}


# Part 3: the original receipt of the real (paid) call, from the receipt book
def original_receipt(conn, model: str, row_id: int) -> dict:
    row = conn.execute(
        "SELECT tokens_in, tokens_out, cost_usd, seconds FROM calls "
        "WHERE model = ? AND row_id = ? AND from_cache = 0 AND valid = 1 "
        "ORDER BY id DESC LIMIT 1",
        (model, row_id),
    ).fetchone()
    return {
        "tokens_in": row[0],
        "tokens_out": row[1],
        "cost_usd": row[2],
        "seconds": row[3],
    }


# Part 4: collect everything for each complaint
test = pd.read_csv("data/test_500.csv").set_index("row_id")
conn = connect()
showcase = []
for row_id, title in SHOWCASE.items():
    complaint = test.loc[row_id, "narrative"]
    answers = {}
    for app_name, model in LLM_MODELS.items():
        saved = get_cached(conn, make_key(model, build_messages(complaint)))
        answers[app_name] = {
            "answer": json.loads(saved),
            "receipt": original_receipt(conn, model, row_id),
        }
    showcase.append(
        {
            "row_id": row_id,
            "title": title,
            "customer_label": test.loc[row_id, "label"],
            "complaint": complaint,
            "answers": answers,
        }
    )
    print(
        f"{row_id}: {title} -> Claude said {answers['Claude Haiku 5.5']['answer']['product']}"
    )

# Part 5: save the museum replays
with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(showcase, file, indent=2)
print(f"Saved {len(showcase)} complaints to {OUTPUT_PATH}")
# NEW CODE END
