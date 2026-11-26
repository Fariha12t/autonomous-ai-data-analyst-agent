import pytest

from ai_data_analyst.sql_guard import UnsafeSQLError, validate_sql


def test_plain_select_gets_limit():
    out = validate_sql("SELECT * FROM sales", max_rows=50)
    assert "LIMIT 50" in out


def test_existing_limit_is_kept():
    assert validate_sql("SELECT * FROM sales LIMIT 5") == "SELECT * FROM sales LIMIT 5"


def test_trailing_semicolon_and_comments_removed():
    out = validate_sql("SELECT 1 -- note\n;", max_rows=10)
    assert ";" not in out and "note" not in out


def test_cte_allowed():
    assert validate_sql("WITH t AS (SELECT 1 AS x) SELECT x FROM t", 10)


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE sales",
        "DELETE FROM sales",
        "UPDATE sales SET revenue = 0",
        "INSERT INTO sales VALUES (1)",
        "SELECT 1; DROP TABLE sales",
        "PRAGMA table_info(sales)",
        "ATTACH DATABASE 'x.db' AS x",
        "SELECT * FROM sales; SELECT 2",
        "",
        "   ",
    ],
)
def test_unsafe_sql_rejected(sql):
    with pytest.raises(UnsafeSQLError):
        validate_sql(sql)


def test_keyword_inside_string_literal_is_fine():
    assert validate_sql("SELECT * FROM sales WHERE region = 'drop; delete'", 10)


def test_column_names_containing_keywords_are_fine():
    assert validate_sql("SELECT created_at, updated_at FROM sales", 10)
