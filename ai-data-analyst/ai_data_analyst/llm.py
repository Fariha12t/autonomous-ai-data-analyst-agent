"""LLM backends.

Every backend implements two operations used by the agent:

* ``plan``      - turn a natural-language question into SQL plus a chart spec
* ``summarize`` - turn query results into an analytical summary with insights

``AnthropicLLM`` and ``OpenAILLM`` call a hosted model. ``OfflineLLM`` is a
rule-based fallback so the project runs end to end without an API key (handy
for demos, CI and tests); it handles common analytical questions only.
"""
from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

from .config import Settings
from .database import TableInfo, schema_to_text


# --------------------------------------------------------------------------- types
@dataclass
class Plan:
    sql: str
    chart: dict = field(default_factory=dict)   # {"type", "x", "y", "series", "title"}
    reasoning: str = ""


class LLMError(RuntimeError):
    pass


# ------------------------------------------------------------------------- prompts
PLAN_SYSTEM = """You are a senior data analyst who writes SQLite SQL.
Given a database schema and a business question, reply with ONE JSON object and nothing else:

{
  "reasoning": "one or two sentences on how you will answer the question",
  "sql": "a single read-only SQLite SELECT query",
  "chart": {"type": "bar|line|pie|scatter|hist|none", "x": "column", "y": "column", "series": "optional column", "title": "short title"}
}

Rules:
- Use only tables and columns that appear in the schema. Never invent columns.
- SQLite dialect: use strftime() for dates, ROUND() for decimals, || for string concatenation.
- Only SELECT (or WITH ... SELECT). One statement. No comments.
- Alias every computed column with a clear snake_case name.
- Aggregate and order the result so it is ready to chart; use LIMIT for top-N questions.
- "x" and "y" in chart must be column names of your query's output. Use "line" for time trends.
- If a previous attempt failed, fix the problem described in the feedback."""

SUMMARY_SYSTEM = """You are a business analyst writing for a non-technical stakeholder.
You are given a question, the SQL that was run, and the result rows.
Write a short analytical summary in Markdown:
1. One-sentence direct answer to the question.
2. 2-4 bullet points of key findings, quoting numbers that appear in the data.
3. A final line starting with "Recommendation:" giving one actionable next step.
Use only numbers present in the data. Do not speculate about causes you cannot see in the data."""


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (tolerates code fences)."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    start = text.find("{")
    if start == -1:
        raise LLMError("Model reply did not contain a JSON object.")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError as exc:
                    raise LLMError(f"Invalid JSON from model: {exc}") from exc
    raise LLMError("Unterminated JSON object in model reply.")


# ------------------------------------------------------------------------ base class
class BaseLLM(ABC):
    name = "base"

    @abstractmethod
    def plan(
        self,
        question: str,
        tables: list[TableInfo],
        history: list[tuple[str, str]] | None = None,
        feedback: str | None = None,
    ) -> Plan: ...

    @abstractmethod
    def summarize(self, question: str, sql: str, df: pd.DataFrame) -> str: ...


class _ChatLLM(BaseLLM):
    """Shared prompt construction for chat-completion style backends."""

    def __init__(self, settings: Settings):
        self.settings = settings

    @abstractmethod
    def _complete(self, system: str, user: str, max_tokens: int) -> str: ...

    def plan(self, question, tables, history=None, feedback=None) -> Plan:
        parts = [f"DATABASE SCHEMA\n{schema_to_text(tables)}"]
        if history:
            recent = "\n".join(f"Q: {q}\nSQL: {s}" for q, s in history[-3:])
            parts.append(f"EARLIER QUESTIONS IN THIS SESSION (for follow-ups)\n{recent}")
        parts.append(f"QUESTION\n{question}")
        if feedback:
            parts.append(f"FEEDBACK ON YOUR PREVIOUS ATTEMPT\n{feedback}")
        data = _extract_json(self._complete(PLAN_SYSTEM, "\n\n".join(parts), 1024))
        sql = str(data.get("sql", "")).strip()
        if not sql:
            raise LLMError("Model returned no SQL.")
        chart = data.get("chart") if isinstance(data.get("chart"), dict) else {}
        return Plan(sql=sql, chart=chart, reasoning=str(data.get("reasoning", "")))

    def summarize(self, question, sql, df) -> str:
        n = self.settings.max_rows_to_llm
        preview = df.head(n).to_csv(index=False)
        note = f"(showing first {n} of {len(df)} rows)\n" if len(df) > n else ""
        user = f"QUESTION\n{question}\n\nSQL\n{sql}\n\nRESULT {note}\n{preview}"
        return self._complete(SUMMARY_SYSTEM, user, 700).strip()


class AnthropicLLM(_ChatLLM):
    name = "anthropic"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise LLMError("Install the SDK first: pip install anthropic") from exc
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    def _complete(self, system, user, max_tokens):
        resp = self.client.messages.create(
            model=self.settings.anthropic_model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


class OpenAILLM(_ChatLLM):
    name = "openai"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMError("Install the SDK first: pip install openai") from exc
        self.client = OpenAI()  # reads OPENAI_API_KEY

    def _complete(self, system, user, max_tokens):
        resp = self.client.chat.completions.create(
            model=self.settings.openai_model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""


# ------------------------------------------------------------------- offline fallback
_AGG_WORDS = {
    "AVG": ("average", "avg", "mean"),
    "COUNT": ("how many", "count", "number of", "no. of"),
    "MAX": ("maximum", "max ", "largest", "biggest"),
    "MIN": ("minimum", "min ", "smallest"),
}
_PREFERRED_MEASURES = ("revenue", "sales", "amount", "total", "profit", "price", "cost", "quantity")
_GRAINS = [
    (("year", "yearly", "annual"), "%Y", "year"),
    (("quarter", "quarterly"), None, "quarter"),
    (("week", "weekly"), "%Y-W%W", "week"),
    (("day", "daily"), "%Y-%m-%d", "day"),
    (("month", "monthly", "trend", "over time", "growth", "seasonal"), "%Y-%m", "month"),
]


def _norm(word: str) -> str:
    return word[:-1] if word.endswith("s") and len(word) > 3 else word


class OfflineLLM(BaseLLM):
    """Rule-based planner/summariser. No network, no API key, limited coverage."""

    name = "offline"

    # ----------------------------------------------------------------- planning
    def plan(self, question, tables, history=None, feedback=None) -> Plan:
        if not tables:
            raise LLMError("No tables are loaded.")
        q = question.lower()
        q_words = {_norm(w) for w in re.findall(r"[a-z]+", q)}
        table = self._pick_table(tables, q_words)
        measures = [c for c in table.columns if c.kind == "numeric"]
        dims = [c for c in table.columns if c.kind == "category"]
        dates = [c for c in table.columns if c.kind == "date"]

        def mentioned(col) -> bool:
            parts = [_norm(p) for p in col.name.lower().split("_")]
            return col.name.lower().replace("_", " ") in q or any(
                p in q_words for p in parts if len(p) >= 4
            )

        dim_cols = [c for c in dims if mentioned(c)]
        picked = [c for c in measures if mentioned(c)][:3]
        measure = picked[0] if picked else None
        if measure is None:
            for pref in _PREFERRED_MEASURES:
                measure = next((c for c in measures if pref in c.name.lower()), None)
                if measure:
                    break
            measure = measure or (measures[0] if measures else None)

        agg = "SUM"
        for func, words in _AGG_WORDS.items():
            if any(w in q for w in words):
                agg = func
                break

        grain = next(((fmt, label) for words, fmt, label in _GRAINS if any(w in q for w in words)), None)
        wants_time = bool(dates) and grain is not None
        top_n = self._top_n(q)
        ascending = any(w in q for w in ("lowest", "worst", "bottom", "least", "smallest"))

        t = f'"{table.name}"'
        if agg == "COUNT" or measure is None:
            expr, alias = "COUNT(*)", "row_count"
            if "order" in q and table.column("order_id"):
                expr, alias = "COUNT(DISTINCT order_id)", "order_count"
            pairs = [(expr, alias)]
        else:
            # "revenue and profit by region" -> one aggregate column per mentioned measure
            pairs = [
                (f'ROUND({agg}("{m.name}"), 2)',
                 f"{agg.lower()}_{m.name}" if agg != "SUM" else f"total_{m.name}")
                for m in (picked or [measure])
            ]
        expr, alias = pairs[0]
        agg_select = ", ".join(f"{e} AS {a}" for e, a in pairs)
        chart_y = alias if len(pairs) == 1 else [a for _, a in pairs]
        chart_title = " & ".join(a.replace("_", " ").title() for _, a in pairs)

        if wants_time:
            d = dates[0].name
            if grain[1] == "quarter":
                period = f"strftime('%Y', \"{d}\") || '-Q' || ((CAST(strftime('%m', \"{d}\") AS INTEGER) + 2) / 3)"
            else:
                period = f"strftime('{grain[0]}', \"{d}\")"
            select = [f"{period} AS period"]
            group = ["period"]
            if dim_cols:
                select.append(f'"{dim_cols[0].name}"')
                group.append(f'"{dim_cols[0].name}"')
            select.append(agg_select)
            sql = (
                f"SELECT {', '.join(select)} FROM {t} GROUP BY {', '.join(group)} ORDER BY period"
            )
            chart = {"type": "line", "x": "period", "y": chart_y, "title": f"{chart_title} by {grain[1]}"}
            if dim_cols:
                chart["series"] = dim_cols[0].name
                chart["y"] = alias  # one measure per series line
            return Plan(sql, chart, f"Aggregating {alias} per {grain[1]} on {dates[0].name}.")

        if dim_cols:
            dim = dim_cols[0].name
            direction = "ASC" if ascending else "DESC"
            limit = f" LIMIT {top_n}" if top_n else ""
            sql = (
                f'SELECT "{dim}", {agg_select} FROM {t} GROUP BY "{dim}" '
                f"ORDER BY {alias} {direction}{limit}"
            )
            chart = {"type": "bar", "x": dim, "y": chart_y, "title": f"{chart_title} by {dim.replace('_', ' ')}"}
            return Plan(sql, chart, f"Grouping {table.name} by {dim} and computing {alias}.")

        if "distribution" in q and measure is not None:
            sql = f'SELECT "{measure.name}" FROM {t}'
            return Plan(sql, {"type": "hist", "x": measure.name, "title": f"Distribution of {measure.name}"},
                        f"Returning raw {measure.name} values for a histogram.")

        if (agg == "COUNT" or measure is not None) and not top_n:
            return Plan(f"SELECT {expr} AS {alias} FROM {t}", {"type": "none"},
                        f"Computing a single overall {alias}.")

        return Plan(f"SELECT * FROM {t} LIMIT 10", {"type": "none"}, "Showing a sample of rows.")

    @staticmethod
    def _pick_table(tables: list[TableInfo], q_words: set[str]) -> TableInfo:
        for t in tables:
            if _norm(t.name.lower()) in q_words:
                return t
        return max(tables, key=lambda t: t.row_count)

    @staticmethod
    def _top_n(q: str) -> int | None:
        m = re.search(r"\b(?:top|bottom|best|worst|first)\s+(\d+)\b", q)
        if m:
            return int(m.group(1))
        if any(w in q for w in ("top", "best", "worst", "bottom", "highest", "lowest")):
            return 5
        return None

    # -------------------------------------------------------------- summarising
    def summarize(self, question, sql, df) -> str:
        if df.empty:
            return "The query returned no rows, so there is nothing to summarise."
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in df.columns if c not in num_cols]
        if len(df) == 1 and len(df.columns) == 1:
            return f"**{df.columns[0].replace('_', ' ').title()}:** {df.iloc[0, 0]:,}" if num_cols else str(df.iloc[0, 0])
        if not num_cols:
            return f"The result has {len(df)} rows and {len(df.columns)} columns."

        y = num_cols[-1]
        bullets: list[str] = []
        label = cat_cols[0] if cat_cols else None
        total = df[y].sum()
        top = df.loc[df[y].idxmax()]
        low = df.loc[df[y].idxmin()]
        head = f"**{y.replace('_', ' ').title()}** across {len(df)} rows"
        series_col = cat_cols[1] if len(cat_cols) > 1 else None

        if label and series_col:
            by_series = df.groupby(series_col)[y].sum().sort_values(ascending=False)
            by_label = df.groupby(label)[y].sum()
            s_top, s_low = by_series.index[0], by_series.index[-1]
            bullets.append(f"{s_top} contributes the most {y} overall: {by_series.iloc[0]:,.2f} ({by_series.iloc[0] / total * 100:.1f}% of {total:,.2f}).")
            if len(by_series) > 1:
                bullets.append(f"{s_low} contributes the least: {by_series.iloc[-1]:,.2f}.")
            bullets.append(f"Strongest {label}: {by_label.idxmax()} ({by_label.max():,.2f}); weakest: {by_label.idxmin()} ({by_label.min():,.2f}).")
            rec = f"Focus on {s_top}'s drivers and check whether {s_low} is under-served."
            head = f"**{y.replace('_', ' ').title()}** by {label} and {series_col}"
        elif label and series_col is None and self._is_time(df[label]):
            first, last = df[y].iloc[0], df[y].iloc[-1]
            change = (last - first) / first * 100 if first else float("nan")
            bullets.append(f"{label.title()} runs from {df[label].iloc[0]} to {df[label].iloc[-1]}; {y} moved from {first:,.2f} to {last:,.2f} ({change:+.1f}%).")
            bullets.append(f"Peak: {top[label]} at {top[y]:,.2f}; lowest: {low[label]} at {low[y]:,.2f}.")
            rec = f"Look into what drove the peak in {top[label]} and whether it can be repeated in weaker periods."
        elif label:
            for m in num_cols[:3]:   # one set of findings per measure column
                m_total, m_top, m_low = df[m].sum(), df.loc[df[m].idxmax()], df.loc[df[m].idxmin()]
                share = m_top[m] / m_total * 100 if m_total else float("nan")
                name = m.replace("_", " ")
                bullets.append(f"Highest {name}: {m_top[label]} at {m_top[m]:,.2f} ({share:.1f}% of the total {m_total:,.2f}); lowest: {m_low[label]} at {m_low[m]:,.2f}.")
            if len(df) > 1 and low[y]:
                bullets.append(f"On {y.replace('_', ' ')}, the top value is {top[y] / low[y]:.1f}x the lowest.")
            rec = f"Review what makes {top[label]} strong and whether {low[label]} has headroom to close the gap."
            if len(num_cols) > 1:
                head = f"**{', '.join(m.replace('_', ' ').title() for m in num_cols[:3])}** by {label}"
        else:
            bullets.append(f"Total {total:,.2f}; maximum {df[y].max():,.2f}; minimum {df[y].min():,.2f}.")
            rec = "Slice this measure by a category or time period to find where the variation comes from."
        return f"{head}.\n\n" + "\n".join(f"- {b}" for b in bullets) + f"\n\nRecommendation: {rec}"

    @staticmethod
    def _is_time(series: pd.Series) -> bool:
        return bool(re.match(r"^\d{4}(-|$)", str(series.iloc[0])))


# ----------------------------------------------------------------------- factory
def build_llm(settings: Settings) -> BaseLLM:
    provider = settings.provider
    if provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise LLMError("ANTHROPIC_API_KEY is not set (see .env.example).")
        return AnthropicLLM(settings)
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise LLMError("OPENAI_API_KEY is not set (see .env.example).")
        return OpenAILLM(settings)
    if provider == "offline":
        return OfflineLLM()
    raise LLMError(f"Unknown provider '{provider}'. Use anthropic, openai or offline.")
