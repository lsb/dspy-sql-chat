"""Text-to-SQL module using DSPy."""
import dspy
from db import create_db


class TextToSQL(dspy.Signature):
    """Translate natural language queries to SQL for the paper_authorships table.

    The paper_authorships table has the following schema:
    - Conference: str (NeurIPS, ICML, or ICLR)
    - Year: int
    - Title: str
    - Author: str
    - Affiliation: str
    """

    natural_language_query: str = dspy.InputField(desc="A natural language question about the paper_authorships table")
    sql_query: str = dspy.OutputField(desc="SQL query that answers the question")


def setup_dspy_ollama():
    """Configure DSPy to use Ollama with qwen3:4b-instruct-2507-q4_K_M."""
    lm = dspy.LM(
        "ollama_chat/qwen3:4b-instruct-2507-q4_K_M",
        api_base="http://localhost:11434",
        api_key="",
        max_tokens=4096  # Ensure enough tokens for thinking/reasoning
    )
    dspy.configure(lm=lm)
    return lm


def translate_to_sql(question: str) -> str:
    """
    Translate a natural language question to SQL.

    Args:
        question: Natural language question about the paper_authorships table

    Returns:
        SQL query string
    """
    # Initialize the predictor
    predictor = dspy.Predict(TextToSQL)

    # Get the SQL translation
    result = predictor(natural_language_query=question)

    return result.sql_query


def query_database(question: str) -> tuple[str, list]:
    """
    Translate a natural language question to SQL and execute it.

    Args:
        question: Natural language question about the paper_authorships table

    Returns:
        Tuple of (sql_query, results)
    """
    # Translate to SQL
    sql_query = translate_to_sql(question)

    # Execute the query
    conn = create_db()
    cursor = conn.cursor()
    cursor.execute(sql_query)
    results = cursor.fetchall()
    conn.close()

    return sql_query, results


if __name__ == "__main__":
    # Set up DSPy with Ollama
    print("Setting up DSPy with Ollama...")
    setup_dspy_ollama()

    # Test with the example from TODO
    test_question = "how many years do we have in the database"
    print(f"\nQuestion: {test_question}")

    sql_query, results = query_database(test_question)
    print(f"Generated SQL: {sql_query}")
    print(f"Result: {results[0][0] if results else 'No results'}")

    # Test with another question
    test_question2 = "how many papers are in the database"
    print(f"\nQuestion: {test_question2}")

    sql_query2, results2 = query_database(test_question2)
    print(f"Generated SQL: {sql_query2}")
    print(f"Result: {results2[0][0] if results2 else 'No results'}")
