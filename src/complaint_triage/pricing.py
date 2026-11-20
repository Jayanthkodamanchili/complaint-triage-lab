# NEW CODE START
"""The price list: US dollars per 1 million tokens, per model (checked 2026-11-17)."""

# Paid list prices = what a bank would really pay.
# llama3.2 runs on our own laptop, so it costs $0.
# Gemini: we use the free tier (we pay $0), but we record the paid price
# so the benchmark is fair. Check the Gemini model name in Step 5.
PRICES = {
    "llama3.2": {"input": 0.00, "output": 0.00},
    "gemini-flash-lite-latest": {"input": 0.25, "output": 1.50},
    "gpt-6-luna": {"input": 0.10, "output": 0.50},
    "gpt-6.2-sol": {"input": 2.00, "output": 10.00},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00},
}


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    """Turn token counts into dollars, using the price list above."""
    price = PRICES[model]
    return (tokens_in * price["input"] + tokens_out * price["output"]) / 1_000_000


if __name__ == "__main__":
    # A typical complaint: about 650 tokens in (rule book + complaint + form), 80 out
    for model in PRICES:
        one = cost_usd(model, 650, 80)
        print(
            f"{model:<26} 1 complaint = ${one:.6f} | 1,000 complaints = ${one * 1000:.2f}"
        )
# NEW CODE END
