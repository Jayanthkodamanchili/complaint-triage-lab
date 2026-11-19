# NEW CODE START
"""Run one LLM on the exam: a 20-complaint dry run, or the full 500."""

import sys
import time

import pandas as pd

from complaint_triage.classify import classify_complaint
from complaint_triage.db import connect
from complaint_triage.providers import PROVIDERS, get_client

# Part 1: settings
OUTPUT_FILES = {  # same folder and naming style as the Phase 2 files
    "ollama": "evals/predictions/llama32_ollama.csv",
    "gemini": "evals/predictions/gemini_flash_lite.csv",
    "openai": "evals/predictions/gpt6_luna.csv",
    "anthropic": "evals/predictions/claude_haiku.csv",
}
GEMINI_PAUSE = 4  # seconds between real Gemini calls, to stay under free-tier limits


# Part 2: read the receipt book for one complaint
def last_receipt(conn, model: str, row_id: int):
    """The newest receipt: was it free from the memory book?"""
    return conn.execute(
        "SELECT from_cache FROM calls WHERE model = ? AND row_id = ? "
        "ORDER BY id DESC LIMIT 1",
        (model, row_id),
    ).fetchone()


def paid_seconds(conn, model: str, row_id: int) -> float:
    """Seconds of the real (not cached) call, so cached answers keep their true speed."""
    row = conn.execute(
        "SELECT seconds FROM calls WHERE model = ? AND row_id = ? "
        "AND from_cache = 0 AND valid = 1 ORDER BY id DESC LIMIT 1",
        (model, row_id),
    ).fetchone()
    if row is None:
        return 0.0
    return row[0]


# Part 3: the exam loop
def run(provider_name: str, mode: str) -> None:
    model = PROVIDERS[provider_name].model
    test = pd.read_csv("data/test_500.csv")
    try:
        get_client(PROVIDERS[provider_name]).models.list()
    except Exception as error:  # noqa: BLE001 - any failure means we cannot start
        print(f"Cannot reach {provider_name} ({type(error).__name__}). Is it running?")
        return
    if mode == "dry":
        # 4 complaints from each of the 5 classes = 20, the same 20 every time
        test = test.groupby("label").sample(4, random_state=42)

    conn = connect()
    first_receipt_id = conn.execute(
        "SELECT COALESCE(MAX(id), 0) FROM calls"
    ).fetchone()[0]
    rows = []

    for number, (_, complaint) in enumerate(test.iterrows(), start=1):
        row_id = int(complaint["row_id"])
        result = classify_complaint(provider_name, complaint["narrative"], row_id, conn)
        if result is None:
            predicted = "INVALID"
        else:
            predicted = result.product
        mark = "OK " if predicted == complaint["label"] else "X  "
        print(
            f"{number:>3}/{len(test)} {mark} {predicted:<17} (answer: {complaint['label']})"
        )

        rows.append(
            {
                "row_id": row_id,
                "label": complaint["label"],
                "predicted": predicted,
                "confidence": "",  # LLMs give no real confidence score
                "seconds": paid_seconds(conn, model, row_id),
            }
        )
        if provider_name == "gemini" and last_receipt(conn, model, row_id)[0] == 0:
            time.sleep(GEMINI_PAUSE)

    # Part 4: the summary for this run
    results = pd.DataFrame(rows)
    correct = (results["predicted"] == results["label"]).sum()
    invalid = (results["predicted"] == "INVALID").sum()
    spent, paid_calls = conn.execute(
        "SELECT COALESCE(SUM(cost_usd), 0), SUM(from_cache = 0) FROM calls "
        "WHERE id > ? AND model = ?",
        (first_receipt_id, model),
    ).fetchone()
    print(f"\n{provider_name}: {correct}/{len(results)} correct, {invalid} invalid")
    print(f"Real calls this run: {paid_calls} | cost this run: ${spent:.4f}")
    print(f"Average seconds per complaint: {results['seconds'].mean():.2f}")

    if mode == "full":
        results.to_csv(OUTPUT_FILES[provider_name], index=False)
        print("Saved:", OUTPUT_FILES[provider_name])


# Part 5: how to run it, e.g.  uv run python -m complaint_triage.run_llm ollama dry
if __name__ == "__main__":
    run(provider_name=sys.argv[1], mode=sys.argv[2])
# NEW CODE END
