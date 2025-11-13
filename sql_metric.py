"""Sophisticated SQL evaluation metrics."""
import sqlite3
from db import create_db
import dspy
from query_timeout import execute_query_with_timeout

def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison (remove extra whitespace, lowercase keywords, etc.)"""
    # Remove extra whitespace
    sql = ' '.join(sql.split())
    # Normalize to uppercase for comparison
    return sql.upper().strip()


def results_match(pred_results, gold_results, pred_sql=None, gold_sql=None):
    """
    Check if results match semantically, allowing for both row and column reordering.

    Returns:
    - 1.0 if results are identical (same order, same columns)
    - 0.999 if results have same data but different row/column order
    - 0.0 if results differ
    """
    # Fast path: exact match
    if pred_results == gold_results:
        return dspy.Prediction(score=1.0, feedback="Exact match")

    # Quick validation
    if not pred_results or not gold_results:
        return dspy.Prediction(score=0.0, feedback="One is empty, other is not")

    if len(pred_results) != len(gold_results):
        return dspy.Prediction(score=0.0, feedback="Different number of rows")

    if len(pred_results[0]) != len(gold_results[0]):
        return dspy.Prediction(score=0.0, feedback="Different number of columns")

    # Not exactly the same, check if semantically equivalent
    try:
        # Check for row reordering: convert to sets of tuples
        if set(pred_results) == set(gold_results):
            return dspy.Prediction(score=0.999, feedback="Row reordering")

        # Check for column reordering: convert each row to frozenset, then compare sets
        if set(frozenset(row) for row in pred_results) == set(frozenset(row) for row in gold_results):
            return dspy.Prediction(score=0.999, feedback="Column reordering")

        # Different results - provide detailed feedback
        pred_sample = pred_results[:3]  # First 3 rows
        gold_sample = gold_results[:3]  # First 3 rows
        
        feedback_parts = ["Different results"]
        if pred_sql and gold_sql:
            feedback_parts.append(f"Predicted SQL: {pred_sql}")
            feedback_parts.append(f"Expected SQL: {gold_sql}")
        
        feedback_parts.append(f"Predicted sample ({len(pred_results)} rows): {pred_sample}")
        feedback_parts.append(f"Expected sample ({len(gold_results)} rows): {gold_sample}")
        
        return dspy.Prediction(score=0.0, feedback=" | ".join(feedback_parts))
    except (TypeError, AttributeError):
        # If comparison fails (unhashable types), not a match
        return dspy.Prediction(score=0.0, feedback="Comparison failed")


def sql_correctness_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """Evaluate predicted SQL correctness with 10-second query timeout.

    Returns a dspy.Prediction with:
    - 1.0 if SQL is identical or produces identical results
    - 0.999 if SQL produces same data with different row/column order
    - 0.001 if SQL executes but produces different results
    - 0.0 if SQL fails to execute or times out
    """
    try:
        # Handle case where prediction might not have sql_query attribute
        if not hasattr(prediction, 'sql_query') or prediction.sql_query is None:
            return dspy.Prediction(score=0.0, feedback="No SQL query in prediction")

        pred_sql = prediction.sql_query
        gold_sql = example.sql_query

        # Check if SQL is identical (after normalization)
        if normalize_sql(pred_sql) == normalize_sql(gold_sql):
            return dspy.Prediction(score=1.0, feedback="Identical SQL")

        conn = create_db()

        try:
            # Execute predicted SQL with timeout
            pred_success, pred_results, pred_error = execute_query_with_timeout(
                conn, pred_sql, timeout_seconds=10.0
            )

            if not pred_success:
                conn.close()
                return dspy.Prediction(score=0.0, feedback=f"Predicted SQL failed: {pred_error}")

            # Execute gold SQL with timeout
            gold_success, gold_results, gold_error = execute_query_with_timeout(
                conn, gold_sql, timeout_seconds=10.0
            )

            if not gold_success:
                conn.close()
                return dspy.Prediction(score=0.0, feedback=f"Gold SQL failed: {gold_error}")

            # Compare results semantically
            return results_match(pred_results, gold_results, pred_sql, gold_sql)

        finally:
            conn.close()
    except Exception as e:
        print(f"Error in metric: {e}")
        return dspy.Prediction(score=0.0, feedback=f"Metric error: {str(e)}")
