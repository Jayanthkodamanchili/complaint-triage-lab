# NEW CODE START
"""Pack our results into one small SQLite file for the Ops Assistant's tools."""

import json
import sqlite3
from pathlib import Path

import pandas as pd

from complaint_triage.db import connect, get_cached, make_key
from complaint_triage.prompts import build_messages

OPS_DB_PATH = "evals/ops.db"
TRIAGE_MODEL = "claude-haiku-5-5"  # the model whose full answer sheets we keep

# Part 1: the 500 exam complaints + Claude's saved answer sheet for each one
test = pd.read_csv("data/test_500.csv")
memory_book = connect()
rows = []
for _, complaint in test.iterrows():
    saved = get_cached(
        memory_book, make_key(TRIAGE_MODEL, build_messages(complaint["narrative"]))
    )
    answer = json.loads(saved)
    rows.append(
        {
            "row_id": complaint["row_id"],
            "customer_label": complaint["label"],
            "customer_issue": complaint["issue"],
            "product": answer["product"],
            "urgency": answer["urgency"],
            "summary": answer["summary"],
        }
    )
complaints = pd.DataFrame(rows)

# Part 2: the leaderboard (Phase 4 report card + the router row)
leaderboard = pd.read_csv("evals/results.csv")
router = {
    "approach": "Router: TF-IDF + Claude",
    "accuracy_%": 88.6,
    "cleaned_acc_%": 95.2,
    "usd_per_1k": 0.043,
    "usd_per_1m": 43,
}
leaderboard = pd.concat([leaderboard, pd.DataFrame([router])], ignore_index=True)

# Part 3: write both tables into a fresh ops.db
Path(OPS_DB_PATH).unlink(missing_ok=True)  # start clean every time
ops = sqlite3.connect(OPS_DB_PATH)
complaints.to_sql("complaints", ops, index=False)
leaderboard.to_sql("leaderboard", ops, index=False)
ops.close()

# Part 4: a quick check
ops = sqlite3.connect(OPS_DB_PATH)
print(
    "complaints:", ops.execute("SELECT COUNT(*) FROM complaints").fetchone()[0], "rows"
)
print(
    "leaderboard:",
    ops.execute("SELECT COUNT(*) FROM leaderboard").fetchone()[0],
    "rows",
)
print(
    "urgency counts:",
    ops.execute("SELECT urgency, COUNT(*) FROM complaints GROUP BY urgency").fetchall(),
)
size_kb = Path(OPS_DB_PATH).stat().st_size / 1000
print(f"Saved {OPS_DB_PATH} ({size_kb:.0f} KB)")
# NEW CODE END
