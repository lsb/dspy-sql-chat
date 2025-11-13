"""SQL evaluation metrics."""
import sqlite3
from db import create_db
import dspy
from query_timeout import execute_query_with_timeout


def sql_correctness_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """Evaluate predicted SQL correctness with 10-second query timeout.

    Returns 1.0 if results match (ignoring row/column order), 0.0 otherwise.
    """
    if not hasattr(prediction, 'sql_query') or prediction.sql_query is None:
        return dspy.Prediction(score=0.0, feedback="No SQL query")

    conn = create_db()
    try:
        pred_results, pred_error = execute_query_with_timeout(
            conn, prediction.sql_query, timeout_seconds=10.0
        )
        if pred_results is None:
            return dspy.Prediction(score=0.0, feedback=f"Predicted SQL failed: {pred_error}")

        gold_results, gold_error = execute_query_with_timeout(
            conn, example.sql_query, timeout_seconds=10.0
        )
        assert gold_results is not None, f"Gold SQL failed: {gold_error}"

        # Compare results: frozenset of frozensets to ignore row and column order
        pred_set = set(frozenset(row) for row in pred_results)
        gold_set = set(frozenset(row) for row in gold_results)

        if pred_set == gold_set:
            return dspy.Prediction(score=1.0, feedback="Match")
        else:
            pred_sample = pred_results[:3]
            gold_sample = gold_results[:3]
            feedback = (
                f"Different results | "
                f"Pred SQL: {prediction.sql_query} | "
                f"Gold SQL: {example.sql_query} | "
                f"Pred rows ({len(pred_results)}): {pred_sample} | "
                f"Gold rows ({len(gold_results)}): {gold_sample}"
            )
            return dspy.Prediction(score=0.0, feedback=feedback)
    finally:
        conn.close()
