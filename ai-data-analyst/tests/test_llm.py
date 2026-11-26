import pytest

from ai_data_analyst.llm import LLMError, OfflineLLM, _extract_json


def test_extract_json_plain_and_fenced():
    assert _extract_json('{"sql": "SELECT 1"}') == {"sql": "SELECT 1"}
    fenced = 'Here you go:\n```json\n{"sql": "SELECT 1", "chart": {"type": "none"}}\n```'
    assert _extract_json(fenced)["chart"]["type"] == "none"


def test_extract_json_nested_braces_in_strings_ok():
    assert _extract_json('{"a": {"b": 1}}') == {"a": {"b": 1}}


def test_extract_json_errors():
    with pytest.raises(LLMError):
        _extract_json("no json here")
    with pytest.raises(LLMError):
        _extract_json('{"sql": ')


def test_offline_group_by_dimension(db):
    plan = OfflineLLM().plan("total revenue by region", db.describe())
    assert 'GROUP BY "region"' in plan.sql and "SUM" in plan.sql
    assert plan.chart["type"] == "bar"
    assert not db.execute(plan.sql).empty


def test_offline_time_trend(db):
    plan = OfflineLLM().plan("monthly revenue trend", db.describe())
    assert "strftime('%Y-%m'" in plan.sql
    assert plan.chart["type"] == "line"
    assert len(db.execute(plan.sql)) == 4


def test_offline_top_n_and_average(db):
    plan = OfflineLLM().plan("top 1 category by average revenue", db.describe())
    assert "AVG" in plan.sql and "LIMIT 1" in plan.sql
    assert len(db.execute(plan.sql)) == 1


def test_offline_count(db):
    plan = OfflineLLM().plan("how many orders are there", db.describe())
    assert db.execute(plan.sql).iloc[0, 0] == 8


def test_offline_summary_groups(db):
    llm = OfflineLLM()
    plan = llm.plan("total revenue by region", db.describe())
    text = llm.summarize("total revenue by region", plan.sql, db.execute(plan.sql))
    assert "South" in text and "Recommendation:" in text


def test_offline_multiple_measures(db):
    llm = OfflineLLM()
    plan = llm.plan("total revenue and quantity by region", db.describe())
    df = db.execute(plan.sql)
    assert {"total_revenue", "total_quantity"} <= set(df.columns)
    assert plan.chart["y"] == ["total_revenue", "total_quantity"]
    assert "quantity" in llm.summarize("q", plan.sql, df)


def test_chat_backend_parses_model_reply_and_builds_prompt(db):
    """The hosted-model path, with the network call replaced by a stub."""
    from ai_data_analyst.config import Settings
    from ai_data_analyst.llm import _ChatLLM

    seen = {}

    class Stub(_ChatLLM):
        name = "stub"

        def _complete(self, system, user, max_tokens):
            seen["user"] = user
            if "RESULT" in user:
                return "  Direct answer.\nRecommendation: do X.  "
            return ('```json\n{"reasoning": "group it", "sql": "SELECT region FROM sales", '
                    '"chart": {"type": "bar", "x": "region", "y": "n"}}\n```')

    llm = Stub(Settings())
    plan = llm.plan("revenue by region", db.describe(), history=[("earlier q", "SELECT 1")],
                    feedback="previous attempt failed")
    assert plan.sql == "SELECT region FROM sales" and plan.chart["type"] == "bar"
    assert "TABLE sales" in seen["user"] and "earlier q" in seen["user"] and "previous attempt failed" in seen["user"]
    assert llm.summarize("q", "SELECT 1", db.execute("SELECT region FROM sales")).startswith("Direct answer.")
