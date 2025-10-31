"""Tests for database module."""
import pytest
import sqlite3
from db import create_db


def test_create_db():
    """Test that database is created successfully."""
    conn = create_db()
    assert isinstance(conn, sqlite3.Connection)
    conn.close()


def test_table_exists():
    """Test that paper_authorships table exists."""
    conn = create_db()
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='paper_authorships'")
    result = cursor.fetchone()

    assert result is not None
    assert result[0] == "paper_authorships"
    conn.close()


def test_distinct_years():
    """Test counting distinct years in the database."""
    conn = create_db()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(DISTINCT year) FROM paper_authorships")
    year_count = cursor.fetchone()[0]

    assert year_count == 19
    conn.close()


def test_total_records():
    """Test total number of records in the database."""
    conn = create_db()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM paper_authorships")
    total_records = cursor.fetchone()[0]

    assert total_records == 172164
    conn.close()


def test_conferences():
    """Test that expected conferences are present."""
    conn = create_db()
    cursor = conn.cursor()

    cursor.execute("SELECT DISTINCT conference FROM paper_authorships ORDER BY conference")
    conferences = [row[0] for row in cursor.fetchall()]

    assert "NeurIPS" in conferences
    assert "ICML" in conferences
    assert "ICLR" in conferences
    assert len(conferences) == 3
    conn.close()


def test_table_schema():
    """Test that table has expected columns."""
    conn = create_db()
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(paper_authorships)")
    columns = [row[1] for row in cursor.fetchall()]

    expected_columns = ["Conference", "Year", "Title", "Author", "Affiliation"]
    assert columns == expected_columns
    conn.close()


def test_sample_data():
    """Test that we can retrieve sample data."""
    conn = create_db()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM paper_authorships LIMIT 1")
    row = cursor.fetchone()

    assert row is not None
    assert len(row) == 5  # Conference, Year, Title, Author, Affiliation
    assert isinstance(row[0], str)  # Conference
    assert isinstance(row[1], int)  # Year
    assert isinstance(row[2], str)  # Title
    assert isinstance(row[3], str)  # Author
    assert isinstance(row[4], str)  # Affiliation
    conn.close()
