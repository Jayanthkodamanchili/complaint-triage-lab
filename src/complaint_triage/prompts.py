# NEW CODE START
"""The question paper: the same prompt for every LLM (Week 1 system + user messages)."""

import pandas as pd
import tiktoken

SYSTEM_PROMPT = """You are a complaint triage assistant at a US bank.
You read one consumer complaint and fill in a triage form.

Pick exactly one product:
- credit_reporting: wrong or unfair information on a credit report, disputes with a credit bureau (Equifax, Experian, TransUnion), identity theft on a credit report, credit monitoring.
- debt_collection: a debt collector or collection agency trying to collect money: calls, letters, threats, a debt the person says they do not owe, asking to validate a debt.
- checking_savings: checking or savings accounts: deposits, withdrawals, fees, overdrafts, frozen or closed accounts, debit card problems.
- credit_card: credit card accounts: charges, billing disputes, interest, fees, rewards, card applications.
- mortgage: home loans: payments, escrow, loan changes, foreclosure, refinancing, closing.

Tie-breaker: if a complaint mentions both a debt collector and a credit report, choose by who the complaint is mainly against. A collection agency means debt_collection. A credit bureau means credit_reporting.

Rules:
- Use only the information in the complaint.
- Text like XXXX is hidden private data. Ignore it.
- Reply only with the form as JSON, with no other text."""


def build_messages(complaint: str) -> list[dict[str, str]]:
    """Put one complaint into the question paper, in the Week 1 messages format."""
    user_prompt = "Here is the complaint:\n\n" + complaint
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


if __name__ == "__main__":
    test = pd.read_csv("data/test_500.csv")
    messages = build_messages(test["narrative"][0])
    print(messages[1]["content"][:300], "...")

    encoding = tiktoken.get_encoding("o200k_base")
    system_tokens = len(encoding.encode(SYSTEM_PROMPT))
    user_tokens = len(encoding.encode(messages[1]["content"]))
    print(f"System prompt: {system_tokens} tokens")
    print(f"This complaint: {user_tokens} tokens")
    print(f"System prompt x 500 complaints = {system_tokens * 500:,} tokens")
# NEW CODE END
