"""Query execution with timeout support for SQL operations."""

import sqlite3
import time
from typing import Optional, Tuple, List, Any


def execute_query_with_timeout(
    conn: sqlite3.Connection,
    query: str,
    timeout_seconds: float = 10.0
) -> Tuple[Optional[List[Any]], Optional[str]]:
    """Execute a SQL query with timeout using SQLite's progress handler.

    Args:
        conn: SQLite database connection
        query: SQL query to execute
        timeout_seconds: Maximum time allowed for query execution (default: 10.0)

    Returns:
        Tuple of (results, error_message). Results is None on error.
    """
    cursor = conn.cursor()
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
        conn.set_progress_handler(None, 0)
        return results, None
    except sqlite3.OperationalError as e:
        conn.set_progress_handler(None, 0)
        if time.time() - start_time > timeout_seconds:
            return None, f"Query timed out after {timeout_seconds:.2f} seconds"
        return None, str(e)
    except Exception as e:
        conn.set_progress_handler(None, 0)
        return None, str(e)

