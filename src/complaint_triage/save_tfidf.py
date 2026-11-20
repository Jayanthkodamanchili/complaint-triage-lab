# NEW CODE START
"""Train TF-IDF + LogReg once and save it to a file, so the app can use it anywhere."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

TRAIN_PATH = "data/train_pool.csv"
MODEL_PATH = "models/tfidf_model.joblib"

# Part 1: the same two pieces and settings as baseline_tfidf.py (Phase 2)
train = pd.read_csv(TRAIN_PATH)
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_features=50_000,
    sublinear_tf=True,
)
model = LogisticRegression(max_iter=1000)

# Part 2: train (learn the word weights from 18,281 complaints)
X_train = vectorizer.fit_transform(train["narrative"])
model.fit(X_train, train["label"])

# Part 3: drop the list of thrown-away words (not needed to predict, and it is big)
vectorizer.stop_words_ = None

# Part 4: save both pieces together in one file
Path("models").mkdir(exist_ok=True)
joblib.dump({"vectorizer": vectorizer, "model": model}, MODEL_PATH)
size_mb = Path(MODEL_PATH).stat().st_size / 1_000_000
print(f"Saved {MODEL_PATH} ({size_mb:.1f} MB)")

# Part 5: check it - load the file back and answer the 500 exam complaints
saved = joblib.load(MODEL_PATH)
test = pd.read_csv("data/test_500.csv")
X_test = saved["vectorizer"].transform(test["narrative"])
predicted = saved["model"].predict(X_test)
accuracy = (predicted == test["label"]).mean()
print(
    f"Loaded back. Accuracy on the 500 exam complaints: {accuracy:.1%} (Phase 2 said 88.0%)"
)
# NEW CODE END
