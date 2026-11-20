# NEW CODE START
"""Phase 4: confusion matrices -> where each approach goes wrong (one picture)."""

import matplotlib

matplotlib.use("Agg")  # draw straight into a file, no pop-up window
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Part 1: the 6 answer sheets (same order as the leaderboard) and short class names
SHEETS = {
    "TF-IDF + LogReg": "tfidf_lr.csv",
    "GPT-6 Luna": "gpt6_luna.csv",
    "Claude Haiku 5.5": "claude_haiku.csv",
    "Gemini Flash-Lite": "gemini_flash_lite.csv",
    "DistilBERT (HF)": "distilbert_hf.csv",
    "llama3.2 (Ollama)": "llama32_ollama.csv",
}
CLASSES = [
    "checking_savings",
    "credit_card",
    "credit_reporting",
    "debt_collection",
    "mortgage",
]
SHORT = ["checking", "card", "credit rep", "debt coll", "mortgage"]
PREDICTIONS_FOLDER = "evals/predictions/"
OUTPUT_PATH = "evals/confusion_matrices.png"


# Part 2: the 3 biggest mix-ups for one approach, in plain words
def top_mixups(matrix, n: int = 3) -> list[str]:
    mixups = []
    for true_index, true_class in enumerate(CLASSES):
        for pred_index, pred_class in enumerate(CLASSES):
            if true_index != pred_index:
                count = matrix[true_index][pred_index]
                mixups.append((count, f"{true_class} -> {pred_class}: {count}"))
    mixups.sort(reverse=True)
    return [text for _, text in mixups[:n]]


# Part 3: one small grid per approach, 2 rows x 3 columns, saved as one picture
if __name__ == "__main__":
    figure, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, (name, file_name) in zip(axes.flatten(), SHEETS.items()):
        sheet = pd.read_csv(PREDICTIONS_FOLDER + file_name)
        matrix = confusion_matrix(sheet["label"], sheet["predicted"], labels=CLASSES)
        ConfusionMatrixDisplay(matrix, display_labels=SHORT).plot(
            ax=ax, cmap="Blues", colorbar=False
        )
        correct = matrix.trace()  # the diagonal = right answers
        ax.set_title(f"{name}: {correct}/500 right")
        ax.set_xlabel("Model said")
        ax.set_ylabel("Customer's label")
        ax.tick_params(axis="x", labelrotation=30)

        print(f"{name}: biggest mix-ups")
        for line in top_mixups(matrix):
            print("   ", line)

    figure.suptitle("Where each approach goes wrong (500 complaints, 100 per class)")
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=120)
    print("\nSaved:", OUTPUT_PATH)
# NEW CODE END
