# NEW CODE START
"""Phase 4: the Label Audit - when the models all disagree with the customer, who is right?"""

import sys
import time
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from complaint_triage.db import connect, get_cached, log_call, make_key, save_cached
from complaint_triage.pricing import cost_usd
from complaint_triage.prompts import SYSTEM_PROMPT
from complaint_triage.providers import PROVIDERS, get_client
from complaint_triage.schemas import Product

JUDGE_MODEL = "claude-sonnet-5-5"  # a stronger model than the 4 contestants
VOTERS = ["tfidf_lr", "gpt6_luna", "claude_haiku", "gemini_flash_lite"]
PREDICTIONS_FOLDER = "evals/predictions/"
OUTPUT_PATH = "evals/label_audit.csv"


# Part 1: the judge's answer sheet (Pydantic, like every LLM answer in this project)
class AuditVerdict(BaseModel):
    verdict: Literal["model_right", "customer_right", "ambiguous"]
    best_product: Product = Field(description="The product you think fits best")
    reason: str = Field(description="One sentence: why")
    summary: str = Field(description="One plain sentence describing the complaint")


# Part 2: find the cases where the "camps" disagree with each other
LLMS = ["gpt6_luna", "claude_haiku", "gemini_flash_lite"]


def find_cases() -> pd.DataFrame:
    test = pd.read_csv("data/test_500.csv")
    for name in VOTERS:
        sheet = pd.read_csv(PREDICTIONS_FOLDER + name + ".csv")
        test[name] = test["row_id"].map(dict(zip(sheet["row_id"], sheet["predicted"])))
    tfidf_right = test["tfidf_lr"] == test["label"]
    llms_wrong = test[LLMS].ne(test["label"], axis=0).sum(axis=1)  # 0 to 3

    # Group A: all 4 approaches disagree with the customer (the first 30 cases)
    # Group B: TF-IDF agrees with the customer, all 3 LLMs disagree
    # Group C: TF-IDF disagrees with the customer, all 3 LLMs agree
    test["group"] = ""
    test.loc[~tfidf_right & (llms_wrong == 3), "group"] = "A_all_disagree"
    test.loc[tfidf_right & (llms_wrong == 3), "group"] = "B_tfidf_with_customer"
    test.loc[~tfidf_right & (llms_wrong == 0), "group"] = "C_llms_with_customer"
    cases = test[test["group"] != ""].copy()

    # the other side's answer = the most common answer that is NOT the customer's
    others = cases[VOTERS].where(cases[VOTERS].ne(cases["label"], axis=0))
    cases["models_said"] = others.mode(axis=1)[0]
    return cases


# Part 3: the judge's question paper (the same 5 product definitions as Phase 3)
DEFINITIONS = SYSTEM_PROMPT.split("Pick exactly one product:")[1].split("Rules:")[0]

JUDGE_PROMPT = f"""You are a senior complaints analyst checking an answer key.
A customer filed a complaint and picked a product category themselves.
Automated systems read the same complaint and picked a different category.
Decide which category is correct, using these definitions:
{DEFINITIONS}
Verdicts:
- model_right: the systems' category clearly fits better than the customer's.
- customer_right: the customer's category clearly fits better.
- ambiguous: the complaint genuinely fits both, or there is not enough information.
Text like XXXX is hidden private data. Reply only with the JSON form."""

schema = AuditVerdict.model_json_schema()
schema["additionalProperties"] = False
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "audit_verdict", "strict": True, "schema": schema},
}


# Part 4: ask the judge about one case (memory book first, then a receipt)
def judge(case, conn) -> AuditVerdict:
    question = (
        f"Customer's choice: {case['label']}\n"
        f"Systems' choice: {case['models_said']}\n\n"
        f"Complaint:\n{case['narrative']}"
    )
    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": question},
    ]
    key = make_key(JUDGE_MODEL, messages)
    saved = get_cached(conn, key)
    if saved is not None:
        return AuditVerdict.model_validate_json(saved)

    client = get_client(PROVIDERS["anthropic"])
    start = time.time()
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=messages,
        response_format=RESPONSE_FORMAT,
        max_tokens=4000,  # room for the judge's hidden thinking + the answer
    )
    seconds = time.time() - start
    answer = response.choices[0].message.content
    verdict = AuditVerdict.model_validate_json(answer)  # the Pydantic referee
    save_cached(conn, key, JUDGE_MODEL, answer)

    tokens_in = response.usage.prompt_tokens
    tokens_out = response.usage.completion_tokens
    log_call(
        conn,
        {
            "model": "judge:" + JUDGE_MODEL,  # kept apart from normal exam receipts
            "row_id": int(case["row_id"]),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": cost_usd(JUDGE_MODEL, tokens_in, tokens_out),
            "seconds": seconds,
            "from_cache": 0,
            "valid": 1,
            "attempts": 1,
            "error": "",
        },
    )
    return verdict


# Part 5: run it - "dry" = first 3 cases only, "full" = all cases + save the CSV
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "dry"
    cases = find_cases()
    print("Cases to judge:", cases["group"].value_counts().sort_index().to_dict())
    if mode == "dry":
        cases = cases.head(3)

    conn = connect()
    rows = []
    for number, (_, case) in enumerate(cases.iterrows(), start=1):
        verdict = judge(case, conn)
        print(
            f"{number:>2}. #{case['row_id']} customer={case['label']:<17} "
            f"models={case['models_said']:<17} -> {verdict.verdict}"
        )
        print(f"    {verdict.reason}")
        rows.append(
            {
                "row_id": case["row_id"],
                "customer_label": case["label"],
                "group": case["group"],
                "models_said": case["models_said"],
                "verdict": verdict.verdict,
                "judge_best_product": verdict.best_product,
                "reason": verdict.reason,
                "summary": verdict.summary,
            }
        )

    results = pd.DataFrame(rows)
    print("\nVerdicts:", results["verdict"].value_counts().to_dict())
    print(pd.crosstab(results["group"], results["verdict"]))
    spent = conn.execute(
        "SELECT SUM(cost_usd) FROM calls WHERE model = ?", ("judge:" + JUDGE_MODEL,)
    ).fetchone()[0]
    print(f"Judge cost so far (all runs): ${spent or 0:.4f}")
    if mode == "full":
        results.to_csv(OUTPUT_PATH, index=False)
        print("Saved:", OUTPUT_PATH)
# NEW CODE END
