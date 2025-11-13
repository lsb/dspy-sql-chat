"""Database module for loading papers.csv into SQLite."""
import sqlite3
import pandas as pd


def create_db(csv_path: str = "papers.csv") -> sqlite3.Connection:
    """
    Create an in-memory SQLite database and load papers.csv into it.
    """
    df = pd.read_csv(csv_path)
    conn = sqlite3.connect(":memory:")
    df.to_sql("paper_authorships", conn, index=False, if_exists="replace")
    return conn
