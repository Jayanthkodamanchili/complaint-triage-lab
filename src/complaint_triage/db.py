# NEW CODE START
"""The memory book (cache) and the receipt book (call log), in one SQLite file."""

import hashlib
import json
import sqlite3
from datetime import datetime

DB_PATH = "data/triage.db"


# Part 1: open the database file and create the two tables (sheets) if missing
def connect(path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            model TEXT,
            answer TEXT,
            saved_at TEXT
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT,
            model TEXT,
            row_id INTEGER,
            tokens_in INTEGER,
            tokens_out INTEGER,
            cost_usd REAL,
            seconds REAL,
            from_cache INTEGER,
            valid INTEGER,
            attempts INTEGER,
            error TEXT
        )"""
    )
    conn.commit()
    return conn


# Part 2: the fingerprint (hash) of "model name + full question"
def make_key(model: str, messages: list[dict[str, str]]) -> str:
    text = model + json.dumps(messages, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Part 3: read from and write to the memory book
def get_cached(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT answer FROM cache WHERE key = ?", (key,)).fetchone()
    if row is None:
        return None
    return row[0]


def save_cached(conn: sqlite3.Connection, key: str, model: str, answer: str) -> None:
    saved_at = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        "INSERT OR REPLACE INTO cache VALUES (?, ?, ?, ?)",
        (key, model, answer, saved_at),
    )
    conn.commit()


# Part 4: write one receipt into the receipt book
def log_call(conn: sqlite3.Connection, receipt: dict) -> None:
    receipt["time"] = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        """INSERT INTO calls (time, model, row_id, tokens_in, tokens_out, cost_usd,
                              seconds, from_cache, valid, attempts, error)
           VALUES (:time, :model, :row_id, :tokens_in, :tokens_out, :cost_usd,
                   :seconds, :from_cache, :valid, :attempts, :error)""",
        receipt,
    )
    conn.commit()


# Part 5: a self-test in a throw-away database that lives only in memory
if __name__ == "__main__":
    conn = connect(":memory:")
    question = [
        {"role": "user", "content": "Complaint #7: my mortgage escrow went up."}
    ]

    gpt_key = make_key("gpt-6-luna", question)
    claude_key = make_key("claude-haiku-5-5", question)
    print("GPT fingerprint:   ", gpt_key[:16], "...")
    print("Claude fingerprint:", claude_key[:16], "...")

    print("GPT in the book before saving?", get_cached(conn, gpt_key))
    save_cached(conn, gpt_key, "gpt-6-luna", '{"product": "mortgage"}')
    print("GPT in the book after saving? ", get_cached(conn, gpt_key))
    print("Claude in the book?           ", get_cached(conn, claude_key))

    receipt = {
        "model": "gpt-6-luna",
        "row_id": 7,
        "tokens_in": 650,
        "tokens_out": 80,
        "cost_usd": 0.000105,
        "seconds": 1.2,
        "from_cache": 0,
        "valid": 1,
        "attempts": 1,
        "error": "",
    }
    log_call(conn, receipt)
    print("Receipts saved:", conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0])
# NEW CODE END
