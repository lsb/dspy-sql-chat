"""Query execution with timeout support for SQL operations.

This module provides utilities to:
1. Measure baseline query execution time
2. Execute queries with configurable timeouts
3. Filter slow queries during dataset loading
"""

import sqlite3
import signal
import time
from typing import Optional, Tuple, List, Any
from db import create_db


class QueryTimeoutError(Exception):
    """Raised when a query execution exceeds the timeout limit."""
    pass


def _timeout_handler(signum, frame):
    """Signal handler for query timeouts."""
    raise QueryTimeoutError("Query execution timed out")


def measure_baseline_query_time(csv_path: str = "papers.csv") -> float:
    """
    Measure the time it takes to execute the baseline query:
    SELECT COUNT(DISTINCT year) FROM paper_authorships

    Args:
        csv_path: Path to the CSV file for database creation

    Returns:
        Time in seconds to execute the baseline query
    """
    conn = create_db(csv_path)
    cursor = conn.cursor()

    baseline_query = "SELECT COUNT(DISTINCT year) FROM paper_authorships"

    start_time = time.time()
    cursor.execute(baseline_query)
    cursor.fetchall()
    elapsed_time = time.time() - start_time

    conn.close()

    return elapsed_time


def execute_query_with_timeout(
    conn: sqlite3.Connection,
    query: str,
    timeout_seconds: Optional[float] = None
) -> Tuple[bool, Optional[List[Any]], Optional[str]]:
    """
    Execute a SQL query with an optional timeout.

    Args:
        conn: SQLite database connection
        query: SQL query to execute
        timeout_seconds: Maximum time allowed for query execution (None = no timeout)

    Returns:
        Tuple of (success, results, error_message):
        - success: True if query completed within timeout, False otherwise
        - results: Query results if successful, None otherwise
        - error_message: Error description if failed, None otherwise
    """
    cursor = conn.cursor()

    if timeout_seconds is None:
        # No timeout, execute normally
        try:
            cursor.execute(query)
            results = cursor.fetchall()
            return True, results, None
        except Exception as e:
            return False, None, str(e)

    # Set up timeout using signal alarm (Unix/Linux only)
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)

    # Set alarm with ceiling to ensure at least 1 second
    signal.alarm(max(1, int(timeout_seconds + 0.5)))

    try:
        cursor.execute(query)
        results = cursor.fetchall()
        signal.alarm(0)  # Cancel the alarm
        signal.signal(signal.SIGALRM, old_handler)  # Restore old handler
        return True, results, None
    except QueryTimeoutError:
        signal.signal(signal.SIGALRM, old_handler)  # Restore old handler
        return False, None, f"Query timed out after {timeout_seconds:.2f} seconds"
    except Exception as e:
        signal.alarm(0)  # Cancel the alarm
        signal.signal(signal.SIGALRM, old_handler)  # Restore old handler
        return False, None, str(e)


def get_query_timeout(baseline_time: float, multiplier: int = 50) -> float:
    """
    Calculate query timeout based on baseline query time.

    Args:
        baseline_time: Time taken by baseline query in seconds
        multiplier: Multiplier for timeout calculation (default: 50)

    Returns:
        Timeout in seconds
    """
    return baseline_time * multiplier


if __name__ == "__main__":
    # Test the timeout functionality
    print("Testing query timeout utilities...")

    # Measure baseline
    print("\n1. Measuring baseline query time...")
    baseline = measure_baseline_query_time()
    print(f"   Baseline query time: {baseline:.4f} seconds")

    # Calculate timeout
    timeout = get_query_timeout(baseline, multiplier=50)
    print(f"   Query timeout (50x baseline): {timeout:.4f} seconds")

    # Test with a simple query
    print("\n2. Testing query execution with timeout...")
    conn = create_db()

    # Fast query
    success, results, error = execute_query_with_timeout(
        conn,
        "SELECT COUNT(*) FROM paper_authorships",
        timeout_seconds=timeout
    )
    print(f"   Fast query: success={success}, results={results}, error={error}")

    conn.close()
    print("\nTests complete!")
