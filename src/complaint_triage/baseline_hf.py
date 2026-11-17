# NEW CODE START
"""Phase 2, approach 2: a pre-built Hugging Face model (DistilBERT) via pipeline().

Picked by hf_compare.py: best of 3 CFPB models on the same 250 practice stories.
"""

import re
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from transformers import pipeline

MODEL_NAME = "agrim-sharma/cfpb-complaints-distilbert"
TEST_PATH = "data/test_500.csv"
OUTPUT_PATH = "evals/predictions/distilbert_hf.csv"

# The model's own label names -> our 5 class names
LABEL_MAP = {
    "retail_banking": "checking_savings",
    "credit_reporting": "credit_reporting",
    "debt_collection": "debt_collection",
    "mortgages_and_loans": "mortgage",
    "credit_card": "credit_card",
}


def clean_text(text: str) -> str:
    """Lowercase and remove punctuation (the model card asks for this)."""
    return re.sub(r"[^\w\s]", " ", text.lower())


def main() -> None:
    test = pd.read_csv(TEST_PATH)
    print(f"Testing {MODEL_NAME} on {len(test)} complaints")

    classifier = pipeline("text-classification", model=MODEL_NAME, device="cpu")
    texts = [clean_text(text) for text in test["narrative"]]

    # Sit the exam (time it)
    start = time.perf_counter()
    outputs = classifier(texts, truncation=True, max_length=512, batch_size=8)
    predict_seconds = time.perf_counter() - start
    print(f"Predicting 500 took {predict_seconds:.1f} seconds")

    # Translate the model's names into our names
    predicted = [LABEL_MAP[output["label"]] for output in outputs]
    confidence = [round(output["score"], 3) for output in outputs]

    accuracy = accuracy_score(test["label"], predicted)
    print(f"Accuracy: {accuracy:.1%}")

    # Save in the same format as tfidf_lr.csv, for Phase 4 scoring
    results = pd.DataFrame(
        {
            "row_id": test["row_id"],
            "label": test["label"],
            "predicted": predicted,
            "confidence": confidence,
            "seconds": predict_seconds / len(test),
        }
    )
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(results)} predictions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
# NEW CODE END
