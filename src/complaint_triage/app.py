# NEW CODE START
"""The Complaint Triage Lab app: 4 tabs in one Gradio page (Week 2 Gradio pattern)."""

import json
import os

import gradio as gr
import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

from complaint_triage.classify import classify_complaint
from complaint_triage.db import connect
from complaint_triage.ops_assistant import DEMO_QUESTIONS, RECORDED_PATH, ask_assistant
from complaint_triage.theme import CSS, FOOTER, GOLD, HEADER, SOFT_TEXT, THEME

load_dotenv(override=True)

# Part 1: demo or live? Live needs BOTH the switch in .env AND an API key
LIVE = os.getenv("APP_MODE") == "live" and bool(os.getenv("ANTHROPIC_API_KEY"))

# Part 2: load the free batsman (TF-IDF) and the museum replays
TFIDF = joblib.load("models/tfidf_model.joblib")
with open("evals/showcase.json", encoding="utf-8") as file:
    SHOWCASE = json.load(file)
THRESHOLD = 0.6  # below this TF-IDF confidence, the router asks Claude (Phase 4)
MAX_CHARACTERS = 4000

# Part 3: the model menu (name shown in the app -> name in providers.py)
ROUTER = "Smart router (TF-IDF first, Claude when unsure)"
TFIDF_ONLY = "TF-IDF only (free, runs here)"
LLM_PROVIDERS = {
    "Claude Haiku 5.5": "anthropic",
    "GPT-6 Luna": "openai",
    "Gemini Flash-Lite": "gemini",
    "llama3.2 (local)": "ollama",
}
MODEL_MENU = [ROUTER, TFIDF_ONLY] + list(LLM_PROVIDERS)

# Part 4: what happens next (product -> team, urgency -> priority)
TEAMS = {
    "credit_reporting": "Credit Reporting team",
    "debt_collection": "Collections Compliance team",
    "checking_savings": "Deposit Accounts team",
    "credit_card": "Card Services team",
    "mortgage": "Mortgage Servicing team",
}
PRIORITY = {
    "high": "High (reply within 24 hours)",
    "medium": "Medium (reply within 3 days)",
    "low": "Low (reply within 10 days)",
}


def next_action(product: str, urgency: str | None) -> str:
    line = f"**Next step:** send to the {TEAMS[product]}"
    if urgency is not None:
        line += f". Priority: {PRIORITY[urgency]}"
    return line + "."


# Part 5: the free batsman answers (product + how sure it is)
def tfidf_guess(complaint: str) -> tuple[str, float]:
    numbers = TFIDF["vectorizer"].transform([complaint])
    chances = TFIDF["model"].predict_proba(numbers)[0]
    best = chances.argmax()
    return TFIDF["model"].classes_[best], float(chances[best])


# Part 6: ask an LLM: a museum replay if we have one, a real call only in live mode
def ask_llm(model_label: str, complaint: str):
    """Returns (answer, receipt text). answer is None when no LLM could answer."""
    for case in SHOWCASE:
        if case["complaint"].strip() == complaint:
            saved = case["answers"][model_label]
            r = saved["receipt"]
            text = (
                f"**Receipt (replay of our real run):** {r['tokens_in']} tokens in, "
                f"{r['tokens_out']} out, cost ${r['cost_usd']:.6f}, took {r['seconds']:.2f} s. "
                "Showing it again today: $0."
            )
            return saved["answer"], text
    if not LIVE:
        return None, (
            f"**{model_label} is switched off in the public demo** (this site holds no "
            "API keys, so it can never spend money). Pick a pre-built complaint to see "
            "its real answers, or choose TF-IDF to classify your own text for free."
        )
    conn = connect()  # open the database inside the function (Gradio uses threads)
    result = classify_complaint(
        LLM_PROVIDERS[model_label], complaint, row_id=0, conn=conn
    )
    tokens_in, tokens_out, cost, seconds, from_cache, error = conn.execute(
        "SELECT tokens_in, tokens_out, cost_usd, seconds, from_cache, error "
        "FROM calls ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if result is None:
        return None, f"**No valid answer after 3 tries.** Last error: {error}"
    if from_cache == 1:
        return result.model_dump(), "**Receipt:** from the memory book (cache): $0."
    return result.model_dump(), (
        f"**Receipt (live call):** {tokens_in} tokens in, {tokens_out} out, "
        f"cost ${cost:.6f}, took {seconds:.2f} s."
    )


# Part 7: what happens when someone clicks "Classify"
def classify_in_app(complaint: str, choice: str):
    complaint = complaint.strip()
    if not complaint:
        return None, "Please type a complaint or pick a pre-built one."
    if len(complaint) > MAX_CHARACTERS:
        return None, f"Please keep the complaint under {MAX_CHARACTERS:,} characters."

    if choice in LLM_PROVIDERS:
        answer, receipt = ask_llm(choice, complaint)
        if answer is None:
            return None, receipt
        return answer, receipt + "\n\n" + next_action(
            answer["product"], answer["urgency"]
        )

    product, sure = tfidf_guess(complaint)
    tfidf_answer = {"product": product, "tfidf_confidence": round(sure, 3)}
    tfidf_note = f"**TF-IDF** (runs on this computer, $0): {product}, {sure:.0%} sure."
    if choice == TFIDF_ONLY or sure >= THRESHOLD:
        if choice == ROUTER:
            tfidf_note += (
                f" That is above {THRESHOLD:.0%}, so the router did not call any LLM."
            )
        return tfidf_answer, tfidf_note + "\n\n" + next_action(product, None)

    # the router: TF-IDF is unsure, so it hands the ball to Claude
    tfidf_note += f" That is below {THRESHOLD:.0%}, so the router asks Claude."
    answer, receipt = ask_llm("Claude Haiku 5.5", complaint)
    if answer is None:
        note = (
            "Claude is switched off in the public demo, so here is TF-IDF's best guess."
        )
        return tfidf_answer, f"{tfidf_note}\n\n{note}\n\n{next_action(product, None)}"
    return answer, (
        f"{tfidf_note}\n\n{receipt}\n\n{next_action(answer['product'], answer['urgency'])}"
    )


# Part 8: the scoreboard (Benchmark tab), read from the Phase 4 report card
matplotlib.use("Agg")  # draw charts in memory, no pop-up window (safe on Windows)
RESULTS = pd.read_csv("evals/results.csv")
ROUTER_ROW = {  # from evals/hybrid.py (Phase 4): TF-IDF + Claude, threshold 0.6
    "approach": "Router: TF-IDF + Claude",
    "accuracy_%": 88.6,
    "cleaned_acc_%": 95.2,
    "usd_per_1k": 0.043,
    "usd_per_1m": 43,
}
ALL_ROWS = pd.concat([RESULTS, pd.DataFrame([ROUTER_ROW])], ignore_index=True)
LEADERBOARD = ALL_ROWS.fillna("n/a").rename(  # n/a = not measured
    columns={
        "approach": "Approach",
        "accuracy_%": "Accuracy % (customer labels)",
        "cleaned_acc_%": "Accuracy % (corrected labels)",
        "macro_f1": "Macro F1",
        "usd_per_1k": "$ per 1K complaints",
        "usd_per_1m": "$ per 1M complaints",
        "median_sec": "Seconds per complaint",
        "valid_json_%": "Valid JSON %",
    }
)
KEY_FINDING = """**Key finding:** on the customers' own labels, TF-IDF wins (88.0%), partly because
it learned from 18,281 complaints labelled by those same customers. Our Label Audit found many
of those labels are wrong. On the corrected labels, the cheap LLMs reach about 95.5% and
TF-IDF 89.6%. **Best value: the router** (TF-IDF first, Claude only when unsure) scores 95.2%
for $43 per million complaints, about 4x cheaper than Claude alone. A premium model
(GPT-6.2 Sol) was no more accurate, at about 11x Claude's cost."""


def accuracy_vs_cost_chart():
    points = ALL_ROWS
    figure, axis = plt.subplots(figsize=(8, 5))
    figure.patch.set_alpha(0)  # see-through, so the night background shows
    axis.set_facecolor("none")
    axis.scatter(points["usd_per_1m"], points["cleaned_acc_%"], s=90, color=GOLD)
    for number, row in points.iterrows():
        shift = 8 if number % 2 == 0 else -16  # above/below, so names don't overlap
        axis.annotate(
            row["approach"],
            (row["usd_per_1m"], row["cleaned_acc_%"]),
            textcoords="offset points",
            xytext=(6, shift),
            color=SOFT_TEXT,
        )
    axis.set_xlabel("Cost per 1 million complaints (US $)")
    axis.set_ylabel("Accuracy on the corrected labels (%)")
    axis.set_title("Accuracy vs cost: top-left is best")
    axis.set_xlim(-10, 320)
    axis.set_ylim(70, 98)
    axis.grid(alpha=0.25)
    for text in [axis.title, axis.xaxis.label, axis.yaxis.label]:
        text.set_color(SOFT_TEXT)
    axis.tick_params(colors=SOFT_TEXT)
    for side in axis.spines.values():
        side.set_color(SOFT_TEXT)
    figure.tight_layout()
    return figure


# Part 9: the third umpire's replays (Label Audit tab), from evals/label_audit.csv
AUDIT = pd.read_csv("evals/label_audit.csv")
WHO_DISAGREED = {  # which models argued against the customer's label
    "A_all_disagree": "All 4 (TF-IDF + 3 LLMs)",
    "B_tfidf_with_customer": "The 3 LLMs (TF-IDF sided with the customer)",
    "C_llms_with_customer": "TF-IDF only (the 3 LLMs sided with the customer)",
}
VERDICT_NAMES = {
    "model_right": "Models right (customer ticked the wrong box)",
    "customer_right": "Customer right (the disagreeing models were wrong)",
    "ambiguous": "Ambiguous (both answers are fair)",
}
AUDIT_TABLE = pd.DataFrame(
    {
        "Complaint (short summary)": AUDIT["summary"],
        "Customer picked": AUDIT["customer_label"],
        "Who disagreed": AUDIT["group"].map(WHO_DISAGREED),
        "They said": AUDIT["models_said"],
        "Judge's verdict": AUDIT["verdict"].map(VERDICT_NAMES),
        "Judge's reason": AUDIT["reason"],
    }
)
COUNTS = AUDIT["verdict"].value_counts()
AUDIT_INTRO = f"""Customers pick the product box themselves when they file a complaint, so some
labels are wrong. For the **{len(AUDIT)} complaints** where our models and the customer disagreed,
a stronger AI judge (Claude Sonnet 5.5) read the full complaint and decided who was right.
**Models right: {COUNTS["model_right"]}** · **Customer right: {COUNTS["customer_right"]}**
· **Ambiguous: {COUNTS["ambiguous"]}**. We also checked 9 verdicts by hand and agreed with all 9.
These corrections make the "corrected labels" accuracy on the Benchmark tab."""


def filter_audit(choice: str) -> pd.DataFrame:
    if choice == "All":
        return AUDIT_TABLE
    return AUDIT_TABLE[AUDIT_TABLE["Judge's verdict"] == choice]


# Part 10: the commentator (Ops Assistant tab): recorded answers, live calls only in live mode
with open(RECORDED_PATH, encoding="utf-8") as file:
    RECORDED = {item["question"].lower(): item for item in json.load(file)}
DEMO_LIST = "\n".join(f"- {question}" for question in DEMO_QUESTIONS)


def ops_chat(message: str, history: list[dict]) -> str:
    key = message.strip().lower()
    if key in RECORDED:
        saved = RECORDED[key]
        tools = ", ".join(saved["tools"]) or "none"
        return f"{saved['answer']}\n\n*Tools used: {tools}. Recorded answer, shown for $0.*"
    if not LIVE:
        return (
            "In the public demo I can only replay these 5 recorded questions, because a "
            "new question needs a paid AI model and this site holds no API keys. "
            f"Copy one into the box:\n\n{DEMO_LIST}"
        )
    clean_history = [{"role": m["role"], "content": m["content"]} for m in history]
    answer, tools_used, cost = ask_assistant(message, clean_history)
    tools = ", ".join(tools_used) or "none"
    return f"{answer}\n\n*Tools used: {tools}. Live answer, cost ${cost:.6f}.*"


# Part 11: the page and its 4 tabs
with gr.Blocks(title="Complaint Triage Lab", theme=THEME, css=CSS) as demo:
    gr.HTML(HEADER)
    if LIVE:
        gr.Markdown(
            "<center>🔓 <b>Live mode:</b> paid models are switched on (owner's laptop).</center>"
        )
    else:
        gr.Markdown(
            "<center>🔒 <b>Public demo:</b> paid AI models are switched off. Pre-built "
            "complaints show the real answers the models gave us; your own text is "
            "answered by the free TF-IDF model.</center>"
        )

    with gr.Tab("🗂️ Classify"):
        gr.Markdown(
            "Type a complaint or click a pre-built one, pick a model, press **Classify**."
        )
        complaint_box = gr.Textbox(
            label="Customer complaint: type your own, or click a pre-built one below",
            placeholder="Example: My credit card was charged twice for one hotel booking",
            lines=6,
        )
        gr.Examples(
            examples=[[case["complaint"]] for case in SHOWCASE],
            example_labels=[case["title"] for case in SHOWCASE],
            inputs=complaint_box,
            label="Pre-built complaints (real CFPB complaints with saved answers)",
        )
        model_menu = gr.Dropdown(
            choices=MODEL_MENU, value=ROUTER, label="Model", filterable=False
        )
        classify_button = gr.Button("Classify", variant="primary")
        answer_box = gr.JSON(label="Answer sheet")
        report_box = gr.Markdown()
        classify_button.click(
            classify_in_app,
            inputs=[complaint_box, model_menu],
            outputs=[answer_box, report_box],
        )

    with gr.Tab("📊 Benchmark"):
        gr.Markdown(KEY_FINDING)
        gr.Dataframe(
            value=LEADERBOARD,
            label="Leaderboard: 6 approaches + the router, 500 complaints",
            interactive=False,  # visitors can read and sort, not edit
        )
        gr.Plot(value=accuracy_vs_cost_chart(), label="Accuracy vs cost")
        gr.Image(
            value="evals/confusion_matrices.png",
            label="Confusion matrices: which products each model mixes up",
        )

    with gr.Tab("🔍 Label Audit"):
        gr.Markdown(AUDIT_INTRO)
        verdict_filter = gr.Radio(
            choices=["All"] + list(VERDICT_NAMES.values()),
            value="All",
            label="Show verdicts",
        )
        audit_table = gr.Dataframe(
            value=AUDIT_TABLE, interactive=False, wrap=True, label="Disputed complaints"
        )
        verdict_filter.change(filter_audit, inputs=verdict_filter, outputs=audit_table)

    with gr.Tab("💬 Ops Assistant"):
        gr.Markdown(
            "Ask about the 500 complaints and the model results. The assistant answers "
            "**only through 4 tools** that look things up in our results database "
            "(get_top_issues, count_by_urgency, compare_models, get_examples), so it "
            "cannot make numbers up. Click an example question to start."
        )
        gr.ChatInterface(
            fn=ops_chat,
            type="messages",
            examples=DEMO_QUESTIONS,
            cache_examples=False,
        )

    gr.Markdown(FOOTER)

# Part 12: start the app when we run this file
if __name__ == "__main__":
    demo.launch(inbrowser=True)
# NEW CODE END
