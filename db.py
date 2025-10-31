"""Database module for loading papers.csv into SQLite."""
import sqlite3
import pandas as pd
from pathlib import Path


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


def test_query(conn: sqlite3.Connection) -> None:
    """Test the database with sample queries."""
    cursor = conn.cursor()

    # Test query: count distinct years
    cursor.execute("SELECT COUNT(DISTINCT year) FROM paper_authorships")
    year_count = cursor.fetchone()[0]
    print(f"Number of distinct years: {year_count}")

    # Test query: total number of records
    cursor.execute("SELECT COUNT(*) FROM paper_authorships")
    total_records = cursor.fetchone()[0]
    print(f"Total number of records: {total_records}")

    # Test query: conferences
    cursor.execute("SELECT DISTINCT conference FROM paper_authorships")
    conferences = [row[0] for row in cursor.fetchall()]
    print(f"Conferences: {', '.join(conferences)}")

    # Test query: sample data
    cursor.execute("SELECT * FROM paper_authorships LIMIT 5")
    print("\nSample data:")
    for row in cursor.fetchall():
        print(row)


if __name__ == "__main__":
    # Create database and run tests
    conn = create_db()
    test_query(conn)
    conn.close()
