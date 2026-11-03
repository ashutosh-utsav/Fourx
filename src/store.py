"""Everything the agent saves, in its own SQLite file (agent.db).

Kept apart from ledger.db on purpose: the ledger is the company's accounting
system (and `python ledger.py reset` deletes its file); this is our own record of
what the agent did. Every write is committed straight away.

Tables:
  conversations  name, which provider it is using
  messages       every message, as JSON in our own provider-neutral format
  approvals      posts waiting for / decided by the clerk. status:
                 waiting -> posting -> posted | failed, or declined | refused
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    name        TEXT PRIMARY KEY,
    provider    TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation  TEXT NOT NULL REFERENCES conversations(name),
    data          TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS approvals (
    id            TEXT PRIMARY KEY,
    conversation  TEXT NOT NULL REFERENCES conversations(name),
    tool_call_id  TEXT NOT NULL,
    invoice_id    TEXT NOT NULL,
    vendor_name   TEXT NOT NULL,
    details       TEXT NOT NULL,
    status        TEXT NOT NULL,
    result        TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
-- At most one open approval per conversation, enforced by the database too.
CREATE UNIQUE INDEX IF NOT EXISTS one_open_approval
    ON approvals(conversation) WHERE status IN ('waiting', 'posting');
"""

OPEN_STATUSES = ("waiting", "posting")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str | Path):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)

    # ------------------------------------------------------------ conversations

    def open_conversation(self, name: str, default_provider: str) -> bool:
        """Create the conversation if needed. Returns True if it is new."""
        with self.conn:
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO conversations (name, provider, created_at) VALUES (?, ?, ?)",
                (name, default_provider, _now()),
            )
        return cur.rowcount == 1

    def get_provider(self, name: str) -> str:
        row = self.conn.execute("SELECT provider FROM conversations WHERE name = ?", (name,)).fetchone()
        return row["provider"]

    def set_provider(self, name: str, provider: str) -> None:
        with self.conn:
            self.conn.execute("UPDATE conversations SET provider = ? WHERE name = ?", (provider, name))

    # ------------------------------------------------------------ messages

    def add_message(self, conversation: str, message: dict) -> None:
        with self.conn:
            self._insert_message(conversation, message)

    def _insert_message(self, conversation: str, message: dict) -> None:
        self.conn.execute(
            "INSERT INTO messages (conversation, data, created_at) VALUES (?, ?, ?)",
            (conversation, json.dumps(message), _now()),
        )

    def messages(self, conversation: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT data FROM messages WHERE conversation = ? ORDER BY id", (conversation,)
        )
        return [json.loads(r["data"]) for r in rows]

    def count_messages(self, conversation: str) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation = ?", (conversation,)
        ).fetchone()[0]

    # ------------------------------------------------------------ approvals

    def add_approval(
        self, conversation: str, tool_call_id: str, invoice_id: str, vendor_name: str, details: dict
    ) -> dict:
        approval_id = uuid.uuid4().hex[:12]
        now = _now()
        with self.conn:
            self.conn.execute(
                "INSERT INTO approvals (id, conversation, tool_call_id, invoice_id, vendor_name,"
                " details, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, 'waiting', ?, ?)",
                (approval_id, conversation, tool_call_id, invoice_id, vendor_name, json.dumps(details), now, now),
            )
        return self.get_approval(approval_id)

    def get_approval(self, approval_id: str) -> dict:
        row = self.conn.execute("SELECT * FROM approvals WHERE id = ?", (approval_id,)).fetchone()
        return self._approval(row)

    def open_approval(self, conversation: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM approvals WHERE conversation = ? AND status IN ('waiting', 'posting')",
            (conversation,),
        ).fetchone()
        return self._approval(row) if row else None

    def approvals_with_status(self, status: str) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM approvals WHERE status = ? ORDER BY created_at", (status,))
        return [self._approval(r) for r in rows]

    def set_approval_status(self, approval_id: str, status: str) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE approvals SET status = ?, updated_at = ? WHERE id = ?", (status, _now(), approval_id)
            )

    def finish_approval(self, approval: dict, status: str, result: dict) -> None:
        """Record the outcome AND the tool result the model will see, in one transaction."""
        with self.conn:
            self.conn.execute(
                "UPDATE approvals SET status = ?, result = ?, updated_at = ? WHERE id = ?",
                (status, json.dumps(result), _now(), approval["id"]),
            )
            self._insert_message(
                approval["conversation"],
                {"role": "tool", "tool_call_id": approval["tool_call_id"], "name": "post_invoice", "result": result},
            )

    @staticmethod
    def _approval(row: sqlite3.Row) -> dict:
        approval = dict(row)
        approval["details"] = json.loads(approval["details"])
        approval["result"] = json.loads(approval["result"]) if approval["result"] else None
        return approval
