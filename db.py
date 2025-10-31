"""Database module for loading papers.csv into SQLite."""
import sqlite3
import pandas as pd


def create_db(csv_path: str = "papers.csv") -> sqlite3.Connection:
    """
    Create an in-memory SQLite database and load papers.csv into it.

    Args:
        csv_path: Path to the papers.csv file

    Returns:
        SQLite connection object with the data loaded
    """
    # Read CSV with pandas
    df = pd.read_csv(csv_path)

    # Create in-memory SQLite database
    conn = sqlite3.connect(":memory:")

    # Write dataframe to SQLite table
    df.to_sql("paper_authorships", conn, index=False, if_exists="replace")

    return conn
