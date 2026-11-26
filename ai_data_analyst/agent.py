"""The analyst agent: question -> SQL -> (self-correct) -> data -> chart -> insights."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import Settings, load_settings
from .database import Database
from .llm import BaseLLM, LLMError, build_llm
from .sql_guard import UnsafeSQLError
from .visualize import make_chart


@dataclass
class Attempt:
    sql: str
    error: str | None = None


@dataclass
class AnalysisResult:
    question: str
    sql: str = ""
    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    chart_path: Path | None = None
    summary: str = ""
    reasoning: str = ""
    attempts: list[Attempt] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_markdown(self, max_rows: int = 25) -> str:
        lines = [f"# {self.question}", ""]
        if not self.ok:
            return "\n".join(lines + [f"**Could not answer:** {self.error}"])
        lines += ["## Summary", "", self.summary, ""]
        if self.chart_path:
            lines += [f"![chart]({Path(self.chart_path).name})", ""]
        lines += ["## SQL", "", "```sql", self.sql, "```", "", "## Data", ""]
        lines.append(self.data.head(max_rows).to_markdown(index=False))
        if len(self.data) > max_rows:
            lines.append(f"\n_Showing {max_rows} of {len(self.data)} rows._")
        return "\n".join(lines) + "\n"


class DataAnalystAgent:
    """Answers natural-language questions about the tables in a `Database`."""

    def __init__(self, db: Database, llm: BaseLLM | None = None, settings: Settings | None = None):
        self.settings = settings or load_settings()
        self.db = db
        self.llm = llm or build_llm(self.settings)
        self.history: list[tuple[str, str]] = []   # (question, sql) pairs for follow-ups

    def ask(self, question: str, save_chart: bool = True) -> AnalysisResult:
        result = AnalysisResult(question=question)
        tables = self.db.describe()
        if not tables:
            result.error = "No data loaded. Load a CSV first."
            return result

        feedback: str | None = None
        plan = None
        for _ in range(self.settings.max_retries):
            try:
                plan = self.llm.plan(question, tables, self.history, feedback)
            except LLMError as exc:
                result.error = f"The model could not produce a query: {exc}"
                return result

            attempt = Attempt(sql=plan.sql)
            result.attempts.append(attempt)
            try:
                df = self.db.execute(plan.sql)
            except UnsafeSQLError as exc:
                attempt.error = f"Rejected by the safety check: {exc}"
            except Exception as exc:  # sqlite3.Error, pandas errors
                attempt.error = f"SQLite error: {exc}"
            else:
                result.sql, result.data, result.reasoning = plan.sql, df, plan.reasoning
                break
            feedback = f"The SQL you wrote was:\n{plan.sql}\nIt failed: {attempt.error}\nWrite a corrected query."
        else:
            result.error = f"Could not produce a working query after {len(result.attempts)} attempts. Last error: {result.attempts[-1].error}"
            return result

        self.history.append((question, result.sql))

        if result.data.empty:
            result.summary = "The query ran but returned no rows."
            return result

        if save_chart:
            out = self.settings.output_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{_slug(question)}.png"
            result.chart_path = make_chart(result.data, plan.chart, out, question)

        try:
            result.summary = self.llm.summarize(question, result.sql, result.data)
        except Exception as exc:  # summary failure should not lose the data
            result.summary = f"_Summary unavailable ({exc})._"
        return result


def _slug(text: str, n: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:n] or "analysis"
