# NEW CODE START
"""Phase 2: find out what LABEL_0 ... LABEL_4 mean in the Hugging Face model.

We use 50 PRACTICE stories per class (never the 500 exam stories).
"""

import pandas as pd
from transformers import pipeline

MODEL_NAME = "Mahesh9/distil-bert-fintuned-cfpb-complaints-data"
TRAIN_PATH = "data/train_pool.csv"
PER_CLASS = 50
SEED = 42


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    sample = train.groupby("label").sample(n=PER_CLASS, random_state=SEED)
    print(f"Checking {len(sample)} practice stories ({PER_CLASS} per class)")

    print(f"Loading {MODEL_NAME} (first run downloads about 270 MB)...")
    classifier = pipeline("text-classification", model=MODEL_NAME, device="cpu")

    outputs = classifier(
        sample["narrative"].tolist(),
        truncation=True,
        max_length=512,
        batch_size=8,
    )
    sample["model_says"] = [output["label"] for output in outputs]

    # Rows = the true class, columns = what the model said
    table = pd.crosstab(sample["label"], sample["model_says"])
    print("\nRows = true class, columns = model's answer:\n")
    print(table)

    print("\nMost common answer per class:")
    for label, row in table.iterrows():
        print(f"  {label:<18} -> {row.idxmax()}  ({row.max()} of {PER_CLASS})")


if __name__ == "__main__":
    main()
# NEW CODE END
