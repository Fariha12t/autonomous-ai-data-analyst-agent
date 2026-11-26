import pandas as pd

from ai_data_analyst import DataAnalystAgent
from ai_data_analyst.llm import BaseLLM, LLMError, Plan
from ai_data_analyst.visualize import infer_spec, make_chart


class ScriptedLLM(BaseLLM):
    """Returns queued plans so the retry loop can be tested without a network."""

    name = "scripted"

    def __init__(self, plans):
        self.plans = list(plans)
        self.feedback_seen: list[str | None] = []

    def plan(self, question, tables, history=None, feedback=None):
        self.feedback_seen.append(feedback)
        if not self.plans:
            raise LLMError("out of plans")
        return self.plans.pop(0)

    def summarize(self, question, sql, df):
        return f"{len(df)} rows"


def test_agent_happy_path_creates_chart(db, settings):
    llm = ScriptedLLM([Plan("SELECT region, SUM(revenue) AS total FROM sales GROUP BY region",
                            {"type": "bar", "x": "region", "y": "total"})])
    res = DataAnalystAgent(db, llm, settings).ask("revenue by region")
    assert res.ok and res.summary == "2 rows"
    assert res.chart_path and res.chart_path.exists()


def test_agent_self_corrects_bad_sql(db, settings):
    llm = ScriptedLLM([
        Plan("SELECT nonexistent FROM sales"),                                   # SQLite error
        Plan("DROP TABLE sales"),                                                # blocked by guard
        Plan("SELECT region, SUM(revenue) AS total FROM sales GROUP BY region"),  # works
    ])
    res = DataAnalystAgent(db, llm, settings).ask("revenue by region")
    assert res.ok and len(res.attempts) == 3
    assert "SQLite error" in llm.feedback_seen[1]
    assert "safety check" in llm.feedback_seen[2]
    assert db.execute("SELECT COUNT(*) AS n FROM sales")["n"][0] == 8   # table untouched


def test_agent_gives_up_after_max_retries(db, settings):
    llm = ScriptedLLM([Plan("SELECT bad FROM sales")] * 5)
    res = DataAnalystAgent(db, llm, settings).ask("anything")
    assert not res.ok and len(res.attempts) == settings.max_retries
    assert "attempts" in res.error


def test_agent_reports_llm_failure(db, settings):
    res = DataAnalystAgent(db, ScriptedLLM([]), settings).ask("anything")
    assert not res.ok and "could not produce a query" in res.error


def test_agent_remembers_history_for_followups(db, settings):
    llm = ScriptedLLM([Plan("SELECT 1 AS x"), Plan("SELECT 2 AS x")])
    agent = DataAnalystAgent(db, llm, settings)
    agent.ask("first")
    agent.ask("second")
    assert [q for q, _ in agent.history] == ["first", "second"]


def test_markdown_report(db, settings):
    res = DataAnalystAgent(db, __import__("ai_data_analyst").llm.OfflineLLM(), settings).ask("total revenue by region")
    md = res.to_markdown()
    assert "## SQL" in md and "## Summary" in md and "| region" in md


def test_infer_spec_variants():
    assert infer_spec(pd.DataFrame({"m": ["2024-01", "2024-02"], "v": [1, 2]}))["type"] == "line"
    assert infer_spec(pd.DataFrame({"c": ["a", "b"], "v": [1, 2]}))["type"] == "bar"
    assert infer_spec(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))["type"] == "scatter"
    assert infer_spec(pd.DataFrame({"v": [1]}))["type"] == "none"


def test_make_chart_falls_back_when_spec_invalid(tmp_path):
    df = pd.DataFrame({"c": ["a", "b", "c"], "v": [3, 1, 2]})
    out = make_chart(df, {"type": "bar", "x": "wrong", "y": "nope"}, tmp_path / "c.png")
    assert out and out.exists()
