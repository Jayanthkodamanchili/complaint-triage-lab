# NEW CODE START
"""Phase 1: build a balanced test set (500) and a training pool from the big parquet file.

The file is too big for 8 GB of RAM, so we read it in small batches, twice:
  Pass 1 - find which rows pass our filters (keep only row number + label)
  Pass 2 - collect the full details of the rows we picked
"""

import os

import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv

load_dotenv(override=True)
PARQUET_PATH = os.getenv("CFPB_PARQUET_PATH")

# Which CFPB product names belong to each of our 5 classes
PRODUCT_TO_LABEL = {
    "Credit reporting, credit repair services, or other personal consumer reports": "credit_reporting",
    "Credit reporting or other personal consumer reports": "credit_reporting",
    "Debt collection": "debt_collection",
    "Checking or savings account": "checking_savings",
    "Credit card": "credit_card",
    "Mortgage": "mortgage",
}

DATE_MIN = pd.Timestamp("2017-05-01")  # first full month with the new product names
DATE_MAX = pd.Timestamp("2024-05-01")  # keep rows BEFORE this date (up to April 2024)
MIN_CHARS = 100  # too short = not enough story to classify
MAX_CHARS = 4000  # too long = expensive to send to GPT / Claude

SEED = 42  # fixed seed: same "random" picks every run
POOL_PER_CLASS = 4000  # rows per class we keep for training + testing
TEST_PER_CLASS = 100  # 5 classes x 100 = 500 test complaints
BATCH_SIZE = 50_000  # rows read at a time (keeps memory low)

TEST_PATH = "data/test_500.csv"
TRAIN_PATH = "data/train_pool.csv"


def find_candidates() -> pd.DataFrame:
    """Pass 1: return the row number and label of every row that passes our filters."""
    parquet_file = pq.ParquetFile(PARQUET_PATH)
    columns = ["date", "product", "consumer_complaint_narrative"]
    candidates = []
    row_offset = 0  # row number of the first row in the current batch

    for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=columns):
        df = batch.to_pandas()
        df["row_id"] = range(row_offset, row_offset + len(df))
        row_offset += len(df)

        df["label"] = df["product"].map(PRODUCT_TO_LABEL)  # NaN if not one of our 5
        lengths = df["consumer_complaint_narrative"].str.len()  # NaN if no story

        keep = (
            df["label"].notna()
            & (df["date"] >= DATE_MIN)
            & (df["date"] < DATE_MAX)
            & (lengths >= MIN_CHARS)
            & (lengths <= MAX_CHARS)
        )
        candidates.append(df.loc[keep, ["row_id", "label"]])
        print(f"  scanned {row_offset:,} rows")

    return pd.concat(candidates, ignore_index=True)


def pick_rows(candidates: pd.DataFrame) -> pd.DataFrame:
    """Randomly pick up to POOL_PER_CLASS rows from each class."""
    picked = []
    for label, group in candidates.groupby("label"):
        how_many = min(POOL_PER_CLASS, len(group))
        picked.append(group.sample(n=how_many, random_state=SEED))
    return pd.concat(picked, ignore_index=True)


def collect_rows(picked: pd.DataFrame) -> pd.DataFrame:
    """Pass 2: read the file again and keep the full details of the picked rows."""
    wanted = set(picked["row_id"])
    parquet_file = pq.ParquetFile(PARQUET_PATH)
    columns = ["date", "product", "issue", "sub_issue", "consumer_complaint_narrative"]
    parts = []
    row_offset = 0

    for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=columns):
        df = batch.to_pandas()
        df["row_id"] = range(row_offset, row_offset + len(df))
        row_offset += len(df)
        parts.append(df[df["row_id"].isin(wanted)])

    rows = pd.concat(parts, ignore_index=True)
    rows["label"] = rows["product"].map(PRODUCT_TO_LABEL)
    rows = rows.rename(columns={"consumer_complaint_narrative": "narrative"})
    output_columns = [
        "row_id",
        "label",
        "product",
        "issue",
        "sub_issue",
        "date",
        "narrative",
    ]
    return rows[output_columns]


def split_test_train(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Take 100 per class for the test set; everything else becomes the training pool."""
    # Same story submitted twice? Keep one copy, so it can't land in both test and train
    rows = rows.drop_duplicates(subset="narrative")

    test_parts = []
    for label, group in rows.groupby("label"):
        test_parts.append(group.sample(n=TEST_PER_CLASS, random_state=SEED))
    test = pd.concat(test_parts, ignore_index=True)

    train = rows[~rows["row_id"].isin(test["row_id"])]
    return test, train


def main() -> None:
    if not PARQUET_PATH:
        print("CFPB_PARQUET_PATH is missing from .env")
        return

    print("Pass 1: finding rows that pass our filters...")
    candidates = find_candidates()
    print("\nCandidates per class (before sampling):")
    print(candidates["label"].value_counts())

    picked = pick_rows(candidates)

    print("\nPass 2: collecting the picked rows...")
    rows = collect_rows(picked)

    test, train = split_test_train(rows)
    test.to_csv(TEST_PATH, index=False)
    train.to_csv(TRAIN_PATH, index=False)

    print(f"\nSaved {len(test)} test rows to {TEST_PATH}")
    print(test["label"].value_counts())
    print(f"\nSaved {len(train)} training rows to {TRAIN_PATH}")
    print(train["label"].value_counts())


if __name__ == "__main__":
    main()
# NEW CODE END
