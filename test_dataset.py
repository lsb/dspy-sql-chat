"""Tests for the question-SQL dataset."""
import json
import pytest
from db import create_db


@pytest.fixture
def dataset():
    """Load the question-SQL pairs dataset."""
    pairs = []
    with open("question_sql_pairs.jsonl", "r") as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def test_dataset_exists(dataset):
    """Test that dataset file exists and has content."""
    assert len(dataset) > 0
    assert len(dataset) >= 150  # Should have at least 150 pairs


def test_dataset_format(dataset):
    """Test that all entries have correct format."""
    for pair in dataset:
        assert "question" in pair
        assert "sql" in pair
        assert isinstance(pair["question"], str)
        assert isinstance(pair["sql"], str)
        assert len(pair["question"]) > 0
        assert len(pair["sql"]) > 0


def test_sql_queries_valid(dataset):
    """Test that all SQL queries execute without errors."""
    conn = create_db()
    cursor = conn.cursor()

    errors = []
    for i, pair in enumerate(dataset):
        try:
            cursor.execute(pair["sql"])
            cursor.fetchall()
        except Exception as e:
            errors.append((i, pair, str(e)))

    conn.close()

    assert len(errors) == 0, f"Found {len(errors)} invalid SQL queries: {errors[:5]}"


def test_question_diversity(dataset):
    """Test that questions are diverse (not too many duplicates)."""
    questions = [pair["question"] for pair in dataset]
    unique_questions = set(questions)

    # At least 80% should be unique
    diversity_ratio = len(unique_questions) / len(questions)
    assert diversity_ratio >= 0.8, f"Only {diversity_ratio*100:.1f}% unique questions"


def test_sql_diversity(dataset):
    """Test that SQL queries are diverse."""
    sql_queries = [pair["sql"] for pair in dataset]
    unique_queries = set(sql_queries)

    # At least 80% should be unique
    diversity_ratio = len(unique_queries) / len(sql_queries)
    assert diversity_ratio >= 0.8, f"Only {diversity_ratio*100:.1f}% unique SQL queries"


def test_query_types_coverage(dataset):
    """Test that dataset covers different query types."""
    sql_queries = [pair["sql"].upper() for pair in dataset]

    # Should have COUNT queries
    count_queries = sum(1 for q in sql_queries if "COUNT" in q)
    assert count_queries > 0, "No COUNT queries found"

    # Should have SELECT DISTINCT queries
    distinct_queries = sum(1 for q in sql_queries if "DISTINCT" in q)
    assert distinct_queries > 0, "No DISTINCT queries found"

    # Should have WHERE clauses
    where_queries = sum(1 for q in sql_queries if "WHERE" in q)
    assert where_queries > 0, "No WHERE queries found"

    # Should have GROUP BY queries
    group_by_queries = sum(1 for q in sql_queries if "GROUP BY" in q)
    assert group_by_queries > 0, "No GROUP BY queries found"

    # Should have JOIN queries
    join_queries = sum(1 for q in sql_queries if "JOIN" in q)
    assert join_queries > 0, "No JOIN queries found"
    assert join_queries >= 10, f"Only {join_queries} JOIN queries, expected at least 10"

    # Should have HAVING clauses
    having_queries = sum(1 for q in sql_queries if "HAVING" in q)
    assert having_queries > 0, "No HAVING queries found"

    # Should have subqueries
    subquery_queries = sum(1 for q in sql_queries if q.count("SELECT") > 1)
    assert subquery_queries > 0, "No subquery queries found"
