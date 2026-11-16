# NEW CODE START
"""Phase 1: a quick look at our 500 test complaints."""

import pandas as pd
import tiktoken

TEST_PATH = "data/test_500.csv"


def main() -> None:
    test = pd.read_csv(TEST_PATH)
    print(f"Test set: {len(test)} complaints\n")

    # 1. How many per class? (should be 100 each)
    print("Complaints per class:")
    print(test["label"].value_counts(), "\n")

    # 2. How long are the stories? (in words)
    test["words"] = test["narrative"].str.split().str.len()
    print("Story length in words, per class:")
    print(test.groupby("label")["words"].describe()[["min", "50%", "max"]], "\n")

    # 3. How many tokens would we send to an LLM? (Week 3: tokenizers)
    encoding = tiktoken.get_encoding("o200k_base")
    test["tokens"] = [len(encoding.encode(text)) for text in test["narrative"]]
    print(f"Total story tokens for all 500: {test['tokens'].sum():,}")
    print(f"Average tokens per story: {test['tokens'].mean():.0f}\n")

    # 4. Most common issues in each class
    print("Top 3 issues per class:")
    for label, group in test.groupby("label"):
        print(f"\n{label}")
        print(group["issue"].value_counts().head(3).to_string())

    # 5. One example story per class (first 300 characters)
    print("\n\nOne example per class:")
    for label, group in test.groupby("label"):
        example = group["narrative"].iloc[0]
        print(f"\n[{label}] {example[:300]}...")


if __name__ == "__main__":
    main()
# NEW CODE END
