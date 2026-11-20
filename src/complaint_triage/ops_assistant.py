# NEW CODE START
"""The Ops Assistant: GPT answers questions ONLY through our 4 tools (Week 2 FlightAI)."""

import json
import sys
import time

from complaint_triage.db import connect, log_call
from complaint_triage.ops_tools import TOOLS, run_tool
from complaint_triage.pricing import cost_usd
from complaint_triage.providers import PROVIDERS, get_client

RECORDED_PATH = "evals/ops_recorded.json"
MODEL = PROVIDERS["openai"].model  # gpt-6-luna: the cheapest model in our test
SYSTEM_PROMPT = """You are the Ops Assistant of the Complaint Triage Lab.
You help a bank's operations team understand 500 real US consumer complaints (CFPB data)
that our triage system sorted into 5 products, and how 6 AI approaches scored on them.
Rules:
- Answer ONLY with facts returned by your tools. Never guess numbers.
- If the tools cannot answer the question, say so politely.
- Keep answers short: 2 to 5 sentences or a short list."""

# Part 1: the 5 questions we record once for the public demo
DEMO_QUESTIONS = [
    "Which approach gives the best accuracy for the money?",
    "How many mortgage complaints are high urgency?",
    "What are the top 3 issues in debt collection complaints?",
    "Show me 2 high-urgency credit card complaints.",
    "Is the most expensive model always the most accurate?",
]


# Part 2: one question -> (answer, list of tools used), with a receipt in the receipt book
def ask_assistant(question: str, history: list[dict] | None = None):
    client = get_client(PROVIDERS["openai"])
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history or []
    messages.append({"role": "user", "content": question})
    tools_used = []
    tokens_in = tokens_out = 0
    start = time.time()

    for _ in range(5):  # at most 5 rounds of tool calls, then stop
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            reasoning_effort="none",
            max_completion_tokens=800,
        )
        tokens_in += response.usage.prompt_tokens
        tokens_out += response.usage.completion_tokens
        message = response.choices[0].message
        if not message.tool_calls:
            answer = message.content
            break
        # the model asked for tools: run each one and hand back the results
        messages.append(message)  # the same pattern as Ed's FlightAI notebook
        for call in message.tool_calls:
            result = run_tool(call.function.name, call.function.arguments)
            tools_used.append(f"{call.function.name}({call.function.arguments})")
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": result}
            )
    else:
        answer = "Sorry, I could not finish that question."

    cost = cost_usd(MODEL, tokens_in, tokens_out)
    log_call(
        connect(),
        {
            "model": "ops:" + MODEL,
            "row_id": 0,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": cost,
            "seconds": time.time() - start,
            "from_cache": 0,
            "valid": 1,
            "attempts": 1,
            "error": "",
        },
    )
    return answer, tools_used, cost


# Part 3: record the 5 demo questions once (run on your laptop, about $0.001)
if __name__ == "__main__":
    questions = DEMO_QUESTIONS if "record" in sys.argv else DEMO_QUESTIONS[:1]
    recorded = []
    total = 0.0
    for question in questions:
        answer, tools_used, cost = ask_assistant(question)
        total += cost
        print(f"\nQ: {question}\nTools: {tools_used}\nA: {answer}\n(${cost:.6f})")
        recorded.append({"question": question, "answer": answer, "tools": tools_used})
    if "record" in sys.argv:
        with open(RECORDED_PATH, "w", encoding="utf-8") as file:
            json.dump(recorded, file, indent=2)
        print(f"\nSaved {len(recorded)} answers to {RECORDED_PATH}. Total ${total:.6f}")
# NEW CODE END
