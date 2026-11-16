"""Phase 0 smoke test: layer 1 lists models ($0), layer 2 makes one tiny call."""

import time  # NEW: to measure seconds

from complaint_triage.providers import PROVIDERS, get_client


def check_models() -> None:
    for provider in PROVIDERS.values():
        try:
            client = get_client(provider)
            ids = [m.id for m in client.models.list()]
            found = any(provider.model in model_id for model_id in ids)
            print(
                f"OK   {provider.name:<10} {len(ids)} models | '{provider.model}' found: {found}"
            )
        except Exception as error:  # noqa: BLE001 - smoke test reports any failure
            print(f"FAIL {provider.name:<10} {type(error).__name__}: {error}")


# NEW CODE START
def check_tiny_call() -> None:
    messages = [{"role": "user", "content": "Reply with just the word OK."}]

    for provider in PROVIDERS.values():
        try:
            client = get_client(provider)
            start = time.time()

            if provider.name == "openai":
                response = client.chat.completions.create(
                    model=provider.model, messages=messages, max_completion_tokens=20
                )
            else:
                response = client.chat.completions.create(
                    model=provider.model, messages=messages, max_tokens=20
                )

            seconds = time.time() - start
            reply = response.choices[0].message.content
            tokens_in = response.usage.prompt_tokens
            tokens_out = response.usage.completion_tokens
            print(
                f"OK   {provider.name:<10} reply={reply!r} | in={tokens_in} out={tokens_out} | {seconds:.1f}s"
            )
        except Exception as error:  # noqa: BLE001 - smoke test reports any failure
            print(f"FAIL {provider.name:<10} {type(error).__name__}: {error}")


if __name__ == "__main__":
    print("--- Layer 1: keys ($0) ---")
    check_models()
    print("--- Layer 2: tiny call ---")
    check_tiny_call()
# NEW CODE END
