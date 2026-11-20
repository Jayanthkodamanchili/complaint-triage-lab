# NEW CODE START
"""Phase 4: the hybrid - TF-IDF answers first, an LLM only when TF-IDF is unsure."""

import sqlite3

import pandas as pd
from score import DB_PATH, cleaned_key, cost_per_complaint  # reuse the examiner
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

THRESHOLD = 0.6  # below this TF-IDF confidence, ask the LLM (chosen in Part 1)
PREDICTIONS_FOLDER = "evals/predictions/"
LLMS = {
    "GPT-6 Luna": ("gpt6_luna.csv", "gpt-6-luna"),
    "Claude Haiku 5.5": ("claude_haiku.csv", "claude-haiku-5-5"),
    "Gemini Flash-Lite": ("gemini_flash_lite.csv", "gemini-flash-lite-latest"),
}


# Part 1: choose the threshold on the PRACTICE pool only (never the 500 exam)
def show_confidence_bands() -> None:
    """How often is TF-IDF right at each confidence level? (5-fold, train_pool)"""
    train = pd.read_csv("data/train_pool.csv")
    model = make_pipeline(
        TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_features=50_000,
            sublinear_tf=True,
        ),
        LogisticRegression(max_iter=1000),
    )
    folds = StratifiedKFold(5, shuffle=True, random_state=42)
    proba = cross_val_predict(
        model, train["narrative"], train["label"], cv=folds, method="predict_proba"
    )
    classes = sorted(train["label"].unique())
    predicted = [classes[i] for i in proba.argmax(axis=1)]
    bands = pd.DataFrame(
        {
            "confidence": proba.max(axis=1),
            "right": [p == t for p, t in zip(predicted, train["label"])],
        }
    )
    bands["band"] = pd.cut(bands["confidence"], [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    table = bands.groupby("band", observed=True)["right"].agg(["count", "mean"])
    table["mean"] = (table["mean"] * 100).round(1)
    print("Practice pool: TF-IDF accuracy % by confidence band")
    print(table.rename(columns={"count": "complaints", "mean": "accuracy_%"}))


# Part 2: build the hybrid answer sheet for one LLM (no new AI calls: saved answers)
def hybrid(llm_file: str) -> pd.DataFrame:
    tfidf = pd.read_csv(PREDICTIONS_FOLDER + "tfidf_lr.csv")
    llm = pd.read_csv(PREDICTIONS_FOLDER + llm_file)
    sheet = tfidf.merge(
        llm[["row_id", "predicted", "seconds"]], on="row_id", suffixes=("", "_llm")
    )
    sheet["asked_llm"] = sheet["confidence"] < THRESHOLD
    sheet["final"] = sheet["predicted"].where(
        ~sheet["asked_llm"], sheet["predicted_llm"]
    )
    return sheet


# Part 3: mark each hybrid like the leaderboard does
if __name__ == "__main__":
    show_confidence_bands()
    print(f"\nRule: ask the LLM when TF-IDF confidence < {THRESHOLD}\n")

    conn = sqlite3.connect(DB_PATH)
    key = cleaned_key()
    rows = []
    for name, (file_name, model) in LLMS.items():
        sheet = hybrid(file_name)
        share = sheet["asked_llm"].mean()
        one_llm_call = cost_per_complaint(conn, model, sheet["row_id"].tolist())
        kept = sheet[sheet["row_id"].isin(key.keys())]
        rows.append(
            {
                "approach": "TF-IDF + " + name,
                "asked_llm_%": round(share * 100, 1),
                "accuracy_%": round((sheet["final"] == sheet["label"]).mean() * 100, 1),
                "cleaned_acc_%": round(
                    (kept["final"] == kept["row_id"].map(key)).mean() * 100, 1
                ),
                "usd_per_1m": round(one_llm_call * share * 1_000_000),
            }
        )
        out = sheet[["row_id", "label", "final", "asked_llm"]]
        out.rename(columns={"final": "predicted"}).to_csv(
            PREDICTIONS_FOLDER + "hybrid_tfidf_" + file_name, index=False
        )

    print(pd.DataFrame(rows).to_string(index=False))
# NEW CODE END
