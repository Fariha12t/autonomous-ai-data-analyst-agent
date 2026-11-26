"""Static checks that make LLM-generated SQL safe to run.

This is the first of two defence layers. The second is a SQLite authorizer
installed in `database.py`, which blocks anything but reads at engine level.
"""
from __future__ import annotations

import re


class UnsafeSQLError(ValueError):
    """Raised when generated SQL is not a single, read-only SELECT."""


FORBIDDEN_KEYWORDS = {
    "insert", "update", "delete", "drop", "alter", "create", "truncate",
    "attach", "detach", "pragma", "vacuum", "reindex", "analyze", "grant", "revoke",
    "begin", "commit", "rollback", "savepoint", "release",
}

_COMMENT_RE = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
_STRING_RE = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")
_LIMIT_RE = re.compile(r"\blimit\s+\d+", re.IGNORECASE)


def _strip_comments(sql: str) -> str:
    return _COMMENT_RE.sub(" ", sql)


def _mask_strings(sql: str) -> str:
    """Blank out string literals so keywords inside them are not flagged."""
    return _STRING_RE.sub("''", sql)


def validate_sql(sql: str, max_rows: int = 500) -> str:
    """Return a cleaned, row-limited version of `sql` or raise UnsafeSQLError."""
    if not sql or not sql.strip():
        raise UnsafeSQLError("Empty SQL.")

    cleaned = _strip_comments(sql).strip()
    cleaned = cleaned.rstrip(";").strip()
    masked = _mask_strings(cleaned)

    if ";" in masked:
        raise UnsafeSQLError("Only a single SQL statement is allowed.")

    first_word = re.match(r"\s*(\w+)", masked)
    if not first_word or first_word.group(1).lower() not in {"select", "with"}:
        raise UnsafeSQLError("Only SELECT queries (optionally starting with WITH) are allowed.")

    words = set(re.findall(r"[a-zA-Z_]+", masked.lower()))
    bad = sorted(words & FORBIDDEN_KEYWORDS)
    if bad:
        raise UnsafeSQLError(f"Forbidden keyword(s) in query: {', '.join(bad)}.")

    if not _LIMIT_RE.search(masked):
        cleaned = f"SELECT * FROM (\n{cleaned}\n) LIMIT {int(max_rows)}"
    return cleaned
