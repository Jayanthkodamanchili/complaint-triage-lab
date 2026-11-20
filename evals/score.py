# NEW CODE START
"""Phase 4: one scorer for every approach -> the leaderboard (evals/results.csv)."""

import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# Part 1: the 6 answer sheets, and the model name each one uses in the receipt book
APPROACHES = {
    "TF-IDF + LogReg": ("tfidf_lr.csv", None),
    "DistilBERT (HF)": ("distilbert_hf.csv", None),
    "llama3.2 (Ollama)": ("llama32_ollama.csv", "llama3.2"),
    "Gemini Flash-Lite": ("gemini_flash_lite.csv", "gemini-flash-lite-latest"),
    "GPT-6 Luna": ("gpt6_luna.csv", "gpt-6-luna"),
    "Claude Haiku 5.5": ("claude_haiku.csv", "claude-haiku-5-5"),
}
PREDICTIONS_FOLDER = "evals/predictions/"
DB_PATH = "data/triage.db"
OUTPUT_PATH = "evals/results.csv"
AUDIT_PATH = "evals/label_audit.csv"


# Part 1b: the cleaned answer key, using the judge's verdicts from the Label Audit
def cleaned_key() -> dict[int, str] | None:
    """Fix labels the judge called model_right; drop the ambiguous ones."""
    if not Path(AUDIT_PATH).exists():
        return None
    key = pd.read_csv("data/test_500.csv").set_index("row_id")["label"].to_dict()
    audit = pd.read_csv(AUDIT_PATH)
    for _, case in audit.iterrows():
        if case["verdict"] == "model_right":
            key[case["row_id"]] = case["judge_best_product"]
        elif case["verdict"] == "ambiguous":
            del key[case["row_id"]]  # no fair answer exists, so leave it out
    return key


# Part 2: average cost of ONE complaint, from the receipt book
def cost_per_complaint(conn, model: str | None, row_ids: list[int]) -> float:
    """Real paid calls only (from_cache = 0, valid = 1), newest receipt per complaint."""
    if model is None:
        return 0.0  # TF-IDF and DistilBERT run on our laptop: $0
    receipts = pd.read_sql(
        "SELECT row_id, cost_usd FROM calls "
        "WHERE model = ? AND from_cache = 0 AND valid = 1 ORDER BY id",
        conn,
        params=(model,),
    )
    receipts = receipts[receipts["row_id"].isin(row_ids)]
    newest = receipts.groupby("row_id").last()  # one receipt per complaint
    return newest["cost_usd"].mean()


# Part 3: mark one answer sheet
def score(name: str, file_name: str, model: str | None, conn, key) -> dict:
    sheet = pd.read_csv(PREDICTIONS_FOLDER + file_name)
    one = cost_per_complaint(conn, model, sheet["row_id"].tolist())
    if model is None:
        valid_json = "n/a"  # these models always return a label, no JSON involved
    else:
        valid_json = round((sheet["predicted"] != "INVALID").mean() * 100, 1)
    accuracy = accuracy_score(sheet["label"], sheet["predicted"])
    macro_f1 = f1_score(sheet["label"], sheet["predicted"], average="macro")
    cleaned = None
    if key is not None:
        kept = sheet[sheet["row_id"].isin(key.keys())]
        cleaned = round((kept["predicted"] == kept["row_id"].map(key)).mean() * 100, 1)
    return {
        "approach": name,
        "accuracy_%": round(accuracy * 100, 1),
        "cleaned_acc_%": cleaned,
        "macro_f1": round(macro_f1, 3),
        "usd_per_1k": round(one * 1_000, 3),
        "usd_per_1m": round(one * 1_000_000),
        "median_sec": round(sheet["seconds"].median(), 4),
        "valid_json_%": valid_json,
    }


# Part 4: mark all 6, sort best first, save and print the leaderboard
if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    key = cleaned_key()
    rows = [score(name, f, model, conn, key) for name, (f, model) in APPROACHES.items()]
    leaderboard = pd.DataFrame(rows).sort_values("accuracy_%", ascending=False)
    leaderboard.to_csv(OUTPUT_PATH, index=False)
    print(leaderboard.to_string(index=False))
    print("\nSaved:", OUTPUT_PATH)
# NEW CODE END
