# NEW CODE START
"""Phase 2, approach 1: TF-IDF + Logistic Regression (classic ML baseline)."""

import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

TRAIN_PATH = "data/train_pool.csv"
TEST_PATH = "data/test_500.csv"
OUTPUT_PATH = "evals/predictions/tfidf_lr.csv"


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    print(f"Training on {len(train):,} complaints, testing on {len(test)}")

    # 1. Turn text into numbers (TF-IDF)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=50_000,
        sublinear_tf=True,
    )
    start = time.perf_counter()
    X_train = vectorizer.fit_transform(train["narrative"])

    # 2. Learn word weights for each class (Logistic Regression)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, train["label"])
    train_seconds = time.perf_counter() - start
    print(f"Training took {train_seconds:.1f} seconds")

    # 3. Predict the 500 test complaints (time it)
    start = time.perf_counter()
    X_test = vectorizer.transform(test["narrative"])
    predicted = model.predict(X_test)
    confidence = model.predict_proba(X_test).max(axis=1)
    predict_seconds = time.perf_counter() - start
    print(f"Predicting 500 took {predict_seconds:.3f} seconds")

    accuracy = accuracy_score(test["label"], predicted)
    print(f"Accuracy: {accuracy:.1%}")

    # 4. Save predictions for Phase 4 scoring
    results = pd.DataFrame(
        {
            "row_id": test["row_id"],
            "label": test["label"],
            "predicted": predicted,
            "confidence": confidence.round(3),
            "seconds": predict_seconds / len(test),
        }
    )
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(results)} predictions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
# NEW CODE END
