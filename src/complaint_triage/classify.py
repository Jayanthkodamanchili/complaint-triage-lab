# NEW CODE START
"""The examiner: send one complaint to one model, check the answer, retry, log it."""

import sys
import time

import pandas as pd
from openai import RateLimitError
from pydantic import ValidationError

from complaint_triage.db import connect, get_cached, log_call, make_key, save_cached
from complaint_triage.pricing import cost_usd
from complaint_triage.prompts import build_messages
from complaint_triage.providers import PROVIDERS, get_client
from complaint_triage.schemas import TriageResult

MAX_ATTEMPTS = 3  # the first try + 2 retries

# Part 1: the printed answer sheet, in the "json_schema" format all 4 providers accept
schema = TriageResult.model_json_schema()
schema["additionalProperties"] = False  # no extra boxes allowed
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "triage_result", "strict": True, "schema": schema},
}


# Part 2: one call to one model (each provider needs its own small settings)
def call_model(provider_name: str, messages: list[dict[str, str]]):
    provider = PROVIDERS[provider_name]
    client = get_client(provider)

    if provider.name == "openai":
        # GPT: as little hidden thinking as allowed (Sol refuses "none", so "low")
        effort = "none" if provider.model == "gpt-6-luna" else "low"
        return client.chat.completions.create(
            model=provider.model,
            messages=messages,
            response_format=RESPONSE_FORMAT,
            max_completion_tokens=2000,
            reasoning_effort=effort,
        )
    if provider_name == "anthropic":
        # Claude: switch off hidden thinking; never send temperature to Claude
        return client.chat.completions.create(
            model=provider.model,
            messages=messages,
            response_format=RESPONSE_FORMAT,
            max_tokens=1000,
            extra_body={"thinking": {"type": "disabled"}},
        )
    if provider_name == "ollama":
        # llama3.2: temperature 0 = always pick the most likely answer
        return client.chat.completions.create(
            model=provider.model,
            messages=messages,
            response_format=RESPONSE_FORMAT,
            max_tokens=1000,
            temperature=0,
        )
    # Gemini: Google advises keeping its default temperature
    return client.chat.completions.create(
        model=provider.model,
        messages=messages,
        response_format=RESPONSE_FORMAT,
        max_tokens=1000,
    )


# Part 3: the examiner (memory book first, then call, check, retry, receipt)
def classify_complaint(
    provider_name: str, complaint: str, row_id: int, conn, verbose: bool = False
) -> TriageResult | None:
    model = PROVIDERS[provider_name].model
    messages = build_messages(complaint)
    key = make_key(model, messages)
    receipt = {
        "model": model,
        "row_id": row_id,
        "tokens_in": 0,
        "tokens_out": 0,
        "cost_usd": 0.0,
        "seconds": 0.0,
        "from_cache": 0,
        "valid": 0,
        "attempts": 0,
        "error": "",
    }

    # Part 3a: is this exact question already in the memory book? Then it's free.
    saved = get_cached(conn, key)
    if saved is not None:
        receipt["from_cache"] = 1
        receipt["valid"] = 1
        log_call(conn, receipt)
        if verbose:
            print("  answer came from the memory book: $0, no call made")
        return TriageResult.model_validate_json(saved)

    # Part 3b: ask the model, up to MAX_ATTEMPTS times
    for attempt in range(1, MAX_ATTEMPTS + 1):
        receipt["attempts"] = attempt
        try:
            start = time.time()
            response = call_model(provider_name, messages)
            receipt["seconds"] += time.time() - start
            receipt["tokens_in"] += response.usage.prompt_tokens
            receipt["tokens_out"] += response.usage.completion_tokens

            answer = response.choices[0].message.content
            result = TriageResult.model_validate_json(answer)  # the Pydantic referee

            if verbose:
                print(f"  model that answered: {response.model}")
                print(f"  finish reason: {response.choices[0].finish_reason}")
            receipt["valid"] = 1
            receipt["error"] = ""
            save_cached(conn, key, model, answer)
            break
        except ValidationError as error:
            receipt["error"] = "invalid answer sheet: " + error.errors()[0]["msg"]
        except RateLimitError:
            receipt["error"] = "rate limited (429)"
            time.sleep(30)  # the provider says "slow down", so wait before retrying
        except Exception as error:  # noqa: BLE001 - log any other failure and retry
            receipt["error"] = f"{type(error).__name__}: {error}"[:300]
            time.sleep(5)
        print(f"  attempt {attempt} failed: {receipt['error']}")

    # Part 3c: write the receipt (also when all attempts failed)
    receipt["cost_usd"] = cost_usd(model, receipt["tokens_in"], receipt["tokens_out"])
    log_call(conn, receipt)
    if receipt["valid"] == 1:
        return result
    return None


# Part 4: self-test on complaint #1 of the exam, asked twice
if __name__ == "__main__":
    provider_name = sys.argv[1] if len(sys.argv) > 1 else "ollama"
    test = pd.read_csv("data/test_500.csv")
    complaint = test["narrative"][0]
    row_id = int(test["row_id"][0])
    conn = connect()

    for round_number in [1, 2]:
        print(f"--- {provider_name}, round {round_number} ---")
        result = classify_complaint(
            provider_name, complaint, row_id, conn, verbose=True
        )
        print("  answer:", result)
        last = conn.execute(
            "SELECT tokens_in, tokens_out, cost_usd, seconds, from_cache, attempts "
            "FROM calls ORDER BY id DESC LIMIT 1"
        ).fetchone()
        print(
            f"  receipt: in={last[0]} out={last[1]} ${last[2]:.6f} "
            f"{last[3]:.1f}s from_cache={last[4]} attempts={last[5]}"
        )
    print("Correct answer (consumer's label):", test["label"][0])
# NEW CODE END
