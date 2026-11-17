# NEW CODE START
"""Phase 2: a mini tournament of pre-built Hugging Face models.

Every model reads the SAME 250 practice stories (50 per class, never the exam).
The model with the best practice score goes on to sit the 500-story exam.
"""

import re
import time

import pandas as pd
from transformers import pipeline

TRAIN_PATH = "data/train_pool.csv"
PER_CLASS = 50
SEED = 42

# Each model's own label names -> our 5 class names
MODELS = {
    "Mahesh9/distil-bert-fintuned-cfpb-complaints-data": {
        "Checking or savings account": "checking_savings",
        "Credit Reporting": "credit_reporting",
        "Debt collection": "debt_collection",
        "Loans / Mortgage": "mortgage",
        "Credit/Prepaid Card": "credit_card",
    },
    "agrim-sharma/cfpb-complaints-distilbert": {
        "retail_banking": "checking_savings",
        "credit_reporting": "credit_reporting",
        "debt_collection": "debt_collection",
        "mortgages_and_loans": "mortgage",
        "credit_card": "credit_card",
    },
    "Solayman-Sameh/consumer-complaint-classifier": {
        "retail_banking": "checking_savings",
        "credit_reporting": "credit_reporting",
        "debt_collection": "debt_collection",
        "mortgages_and_loans": "mortgage",
        "credit_card": "credit_card",
    },
}


def clean_text(text: str) -> str:
    """Lowercase and remove punctuation (agrim-sharma's card asks for this)."""
    return re.sub(r"[^\w\s]", " ", text.lower())


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    sample = train.groupby("label").sample(n=PER_CLASS, random_state=SEED)
    print(f"Every model reads the same {len(sample)} practice stories\n")

    scores = []
    for model_name, label_map in MODELS.items():
        print(f"--- {model_name} ---")
        classifier = pipeline("text-classification", model=model_name, device="cpu")

        texts = sample["narrative"].tolist()
        if model_name.startswith("agrim-sharma"):
            texts = [clean_text(text) for text in texts]

        start = time.perf_counter()
        outputs = classifier(texts, truncation=True, max_length=512, batch_size=8)
        seconds = time.perf_counter() - start

        sample["predicted"] = [label_map[output["label"]] for output in outputs]
        correct = sample["label"] == sample["predicted"]
        print(f"Correct: {correct.sum()} of {len(sample)} ({correct.mean():.1%})")
        print("Correct per class (out of 50):")
        print(correct.groupby(sample["label"]).sum().to_string(), "\n")

        scores.append(
            {
                "model": model_name,
                "accuracy": round(correct.mean(), 3),
                "seconds": round(seconds, 1),
            }
        )

    print("=== Practice leaderboard ===")
    leaderboard = pd.DataFrame(scores).sort_values("accuracy", ascending=False)
    print(leaderboard.to_string(index=False))


if __name__ == "__main__":
    main()
# NEW CODE END
