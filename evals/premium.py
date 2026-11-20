# NEW CODE START
"""Phase 4: the premium test - is a pricier model worth it? (same 100 complaints)"""

import sqlite3

import pandas as pd
from score import APPROACHES, DB_PATH, cleaned_key, cost_per_complaint

PREDICTIONS_FOLDER = "evals/predictions/"
PREMIUM = ("GPT-6.2 Sol (premium)", "gpt62_sol_premium.csv", "gpt-6.2-sol")

# Part 1: every approach, marked on ONLY the 100 complaints the premium model sat
if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    key = cleaned_key()
    premium_ids = pd.read_csv(PREDICTIONS_FOLDER + PREMIUM[1])["row_id"]

    contestants = {name: (f, model) for name, (f, model) in APPROACHES.items()}
    contestants[PREMIUM[0]] = (PREMIUM[1], PREMIUM[2])

    rows = []
    for name, (file_name, model) in contestants.items():
        sheet = pd.read_csv(PREDICTIONS_FOLDER + file_name)
        sheet = sheet[sheet["row_id"].isin(premium_ids)]
        kept = sheet[sheet["row_id"].isin(key.keys())]
        one = cost_per_complaint(conn, model, sheet["row_id"].tolist())
        rows.append(
            {
                "approach": name,
                "complaints": len(sheet),
                "accuracy_%": round(
                    (sheet["predicted"] == sheet["label"]).mean() * 100, 1
                ),
                "cleaned_acc_%": round(
                    (kept["predicted"] == kept["row_id"].map(key)).mean() * 100, 1
                ),
                "usd_per_1m": round(one * 1_000_000),
                "median_sec": round(sheet["seconds"].median(), 2),
            }
        )

    table = pd.DataFrame(rows).sort_values("cleaned_acc_%", ascending=False)
    print(table.to_string(index=False))
# NEW CODE END
