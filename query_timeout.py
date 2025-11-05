"""Query execution with timeout support for SQL operations.

This module provides utilities to:
1. Measure baseline query execution time
2. Execute queries with configurable timeouts
3. Filter slow queries during dataset loading

Uses SQLite's progress handler for thread-safe timeout implementation.
"""

import sqlite3
import time
from typing import Optional, Tuple, List, Any
from db import create_db


class QueryTimeoutError(Exception):
    """Raised when a query execution exceeds the timeout limit."""
    pass


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
    Execute a SQL query with an optional timeout using SQLite's progress handler.

    This implementation is thread-safe and works in any thread, not just the main thread.
    SQLite's progress handler is called periodically during query execution, allowing
    us to check if the timeout has been exceeded.

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

    # Thread-safe timeout using SQLite's progress handler
    start_time = time.time()

    def progress_handler():
        """Called periodically by SQLite during query execution."""
        if time.time() - start_time > timeout_seconds:
            # Returning non-zero causes SQLite to abort the query
            return 1
        return 0

    # Set progress handler to be called every N virtual machine instructions
    # Lower N = more frequent checks but more overhead. 1000 is a reasonable balance.
    conn.set_progress_handler(progress_handler, 1000)

    try:
        cursor.execute(query)
        results = cursor.fetchall()
        # Clear the progress handler
        conn.set_progress_handler(None, 0)
        return True, results, None
    except sqlite3.OperationalError as e:
        # Clear the progress handler
        conn.set_progress_handler(None, 0)
        # Check if it was our timeout that caused the error
        if time.time() - start_time > timeout_seconds:
            return False, None, f"Query timed out after {timeout_seconds:.2f} seconds"
        return False, None, str(e)
    except Exception as e:
        # Clear the progress handler
        conn.set_progress_handler(None, 0)
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
