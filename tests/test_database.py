import pytest

from ai_data_analyst import Database
from ai_data_analyst.database import sanitize_identifier, schema_to_text
from ai_data_analyst.sql_guard import UnsafeSQLError


def test_sanitize_identifier():
    assert sanitize_identifier("Order Date (UTC)") == "order_date_utc"
    assert sanitize_identifier("2024 sales") == "t_2024_sales"
    assert sanitize_identifier("???") == "col"


def test_describe_infers_column_kinds(db):
    (table,) = db.describe()
    kinds = {c.name: c.kind for c in table.columns}
    assert table.row_count == 8
    assert kinds["order_id"] == "id"
    assert kinds["order_date"] == "date"
    assert kinds["region"] == "category"
    assert kinds["revenue"] == "numeric"


def test_schema_text_mentions_table_and_columns(db):
    text = schema_to_text(db.describe())
    assert "TABLE sales (8 rows)" in text and "revenue" in text


def test_execute_returns_dataframe(db):
    out = db.execute("SELECT region, SUM(revenue) AS total FROM sales GROUP BY region ORDER BY region")
    assert list(out["region"]) == ["North", "South"]
    assert out["total"].sum() == pytest.approx(1480.0)


def test_row_cap_applied(df):
    small = Database(max_rows=3)
    small.load_dataframe(df, "sales")
    assert len(small.execute("SELECT * FROM sales")) == 3


def test_writes_blocked_by_guard(db):
    with pytest.raises(UnsafeSQLError):
        db.execute("DELETE FROM sales")


def test_writes_blocked_at_engine_level(db):
    """Even if the guard were bypassed, the SQLite authorizer refuses writes."""
    import sqlite3

    with pytest.raises(sqlite3.DatabaseError):
        db.conn.execute("DELETE FROM sales")
    assert db.execute("SELECT COUNT(*) AS n FROM sales")["n"][0] == 8


def test_load_csv(tmp_path, df):
    path = tmp_path / "My Orders.csv"
    df.to_csv(path, index=False)
    database = Database()
    assert database.load_csv(path) == "my_orders"
    assert database.table_names() == ["my_orders"]
