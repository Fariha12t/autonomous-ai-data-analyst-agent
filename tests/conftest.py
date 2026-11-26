import pandas as pd
import pytest

from ai_data_analyst import Database
from ai_data_analyst.config import Settings


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "order_id": [f"O{i}" for i in range(1, 9)],
            "order_date": ["2024-01-05", "2024-01-20", "2024-02-03", "2024-02-18",
                           "2024-03-02", "2024-03-15", "2024-03-28", "2024-04-01"],
            "region": ["North", "South", "North", "South", "North", "South", "North", "South"],
            "category": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "revenue": [100.0, 200.0, 150.0, 250.0, 120.0, 220.0, 180.0, 260.0],
            "quantity": [1, 2, 1, 3, 1, 2, 2, 3],
        }
    )


@pytest.fixture
def db(df):
    database = Database(max_rows=100)
    database.load_dataframe(df, "sales")
    return database


@pytest.fixture
def settings(tmp_path):
    return Settings(provider="offline", output_dir=tmp_path, max_retries=3)
