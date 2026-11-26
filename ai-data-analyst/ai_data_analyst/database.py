"""SQLite-backed data layer: load CSVs, describe the schema, run read-only queries."""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .sql_guard import validate_sql

_DATE_RE = re.compile(r"^\d{4}-\d{2}(-\d{2})?")


@dataclass
class ColumnInfo:
    name: str
    sql_type: str
    kind: str                     # numeric | date | category | id | text
    samples: list = field(default_factory=list)


@dataclass
class TableInfo:
    name: str
    row_count: int
    columns: list[ColumnInfo]

    def column(self, name: str) -> ColumnInfo | None:
        return next((c for c in self.columns if c.name == name), None)


def sanitize_identifier(name: str) -> str:
    """Turn a file name or column header into a safe snake_case SQL identifier."""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()
    if not s:
        s = "col"
    if s[0].isdigit():
        s = f"t_{s}"
    return s


def schema_to_text(tables: list[TableInfo]) -> str:
    """Render the schema as compact text for LLM prompts."""
    lines: list[str] = []
    for t in tables:
        lines.append(f"TABLE {t.name} ({t.row_count} rows)")
        for c in t.columns:
            sample = ", ".join(repr(s) for s in c.samples[:4])
            lines.append(f"  - {c.name} {c.sql_type} [{c.kind}]  e.g. {sample}")
    return "\n".join(lines)


class Database:
    """A single SQLite connection with CSV loading and guarded, read-only querying."""

    def __init__(self, path: str | Path = ":memory:", max_rows: int = 500):
        self.path = str(path)
        self.max_rows = max_rows
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self._locked = False

    # ---------------------------------------------------------------- loading
    def load_dataframe(self, df: pd.DataFrame, table: str) -> str:
        df = df.copy()
        df.columns = [sanitize_identifier(c) for c in df.columns]
        # parse date-looking text columns so they are stored as ISO strings
        for col in df.columns:
            if df[col].dtype == object and df[col].dropna().astype(str).head(20).map(
                lambda v: bool(_DATE_RE.match(v))
            ).all() and df[col].notna().any():
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() > 0.9:
                    df[col] = parsed.dt.strftime("%Y-%m-%d")
        table = sanitize_identifier(table)
        self._unlock()
        try:
            df.to_sql(table, self.conn, if_exists="replace", index=False)
            self.conn.commit()
        finally:
            self._lock()
        return table

    def load_csv(self, path: str | Path, table: str | None = None) -> str:
        path = Path(path)
        df = pd.read_csv(path)
        return self.load_dataframe(df, table or path.stem)

    # ------------------------------------------------------------ introspection
    def table_names(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        return [r[0] for r in rows]

    def describe(self) -> list[TableInfo]:
        self._unlock()  # PRAGMA is needed for introspection; re-locked below
        try:
            return self._describe()
        finally:
            self._lock()

    def _describe(self) -> list[TableInfo]:
        tables: list[TableInfo] = []
        for name in self.table_names():
            row_count = self.conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            cols: list[ColumnInfo] = []
            for _, cname, ctype, *_ in self.conn.execute(f'PRAGMA table_info("{name}")').fetchall():
                samples = [
                    r[0]
                    for r in self.conn.execute(
                        f'SELECT DISTINCT "{cname}" FROM "{name}" WHERE "{cname}" IS NOT NULL LIMIT 4'
                    ).fetchall()
                ]
                n_distinct = self.conn.execute(
                    f'SELECT COUNT(DISTINCT "{cname}") FROM "{name}"'
                ).fetchone()[0]
                cols.append(
                    ColumnInfo(cname, ctype or "TEXT", self._kind(cname, ctype, samples, n_distinct, row_count), samples)
                )
            tables.append(TableInfo(name, row_count, cols))
        return tables

    @staticmethod
    def _kind(name: str, sql_type: str, samples: list, n_distinct: int, row_count: int) -> str:
        lname = name.lower()
        is_numeric_type = (sql_type or "").upper() in {"INTEGER", "REAL", "NUMERIC", "FLOAT", "DOUBLE", "INT"}
        if lname == "id" or lname.endswith("_id") or lname.endswith("_key"):
            return "id"
        if samples and all(isinstance(s, str) and _DATE_RE.match(s) for s in samples):
            return "date"
        if is_numeric_type:
            return "numeric"
        if row_count and n_distinct <= max(25, row_count * 0.05):
            return "category"
        return "text"

    # --------------------------------------------------------------- querying
    def execute(self, sql: str) -> pd.DataFrame:
        """Validate and run a read-only query; returns a DataFrame."""
        safe_sql = validate_sql(sql, self.max_rows)
        return pd.read_sql_query(safe_sql, self.conn)

    # ------------------------------------------------- engine-level protection
    def _lock(self) -> None:
        """Install an authorizer so only reads can happen, whatever SQL is sent."""
        allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
        recursive = getattr(sqlite3, "SQLITE_RECURSIVE", None)
        if recursive is not None:
            allowed.add(recursive)

        def authorizer(action, *_):
            return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY

        self.conn.set_authorizer(authorizer)
        self._locked = True

    def _unlock(self) -> None:
        self.conn.set_authorizer(None)
        self._locked = False

    def close(self) -> None:
        self.conn.close()
