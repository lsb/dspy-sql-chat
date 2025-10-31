"""Tests for text-to-SQL module."""
import pytest
from text_to_sql import setup_dspy_ollama, translate_to_sql, query_database


@pytest.fixture(scope="module")
def dspy_setup():
    """Set up DSPy once for all tests."""
    setup_dspy_ollama()


def test_translate_years_query(dspy_setup):
    """Test translating 'how many years' question to SQL."""
    question = "how many years do we have in the database"
    sql = translate_to_sql(question)

    # The SQL should contain COUNT and DISTINCT and year
    assert "COUNT" in sql.upper()
    assert "DISTINCT" in sql.upper()
    assert "YEAR" in sql.upper()


def test_query_years(dspy_setup):
    """Test querying for distinct years."""
    question = "how many years do we have in the database"
    sql_query, results = query_database(question)

    # Should return 19 years
    assert len(results) > 0
    assert results[0][0] == 19


def test_query_total_records(dspy_setup):
    """Test querying for total records."""
    question = "how many papers are in the database"
    sql_query, results = query_database(question)

    # Should return 172164 records
    assert len(results) > 0
    assert results[0][0] == 172164
