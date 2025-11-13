"""Text-to-SQL module using DSPy."""
import os
import dspy
from db import create_db


# Model configuration
OLLAMA_BASE_MODEL = os.environ.get("OLLAMA_BASE_MODEL", "ollama_chat/gpt-oss:20b")
OLLAMA_API_BASE = "http://localhost:11434"
MAX_TOKENS = 4096  # Ensure enough tokens for thinking/reasoning


class TextToSQL(dspy.Signature):
    """Translate natural language queries to SQL for the paper_authorships table.
    There is one row for each paper-author pair: a paper with N authors has N corresponding rows.
    This is a read-only database; all updates, inserts, and deletes should be rejected with `select 'database is read-only'`.
    Reject inappropriate queries with `select 'query violates content policy'`.

    The paper_authorships table has the following schema:
    - Conference: str (NeurIPS, ICML, or ICLR)
    - Year: int
    - Title: str
    - Author: str
    - Affiliation: str
    """

    natural_language_query: str = dspy.InputField(desc="A natural language question about the paper_authorships table")
    sql_query: str = dspy.OutputField(desc="A clean SQL query that answers the question. Output only the SQL query itself, with no markdown formatting, no code blocks, no extra text, and no completion markers. Do not include semicolons at the end. The query should be a single valid SQL statement ready to execute.")


def setup_dspy_ollama():
    """Configure DSPy to use Ollama with the configured model."""
    lm = dspy.LM(
        OLLAMA_BASE_MODEL,
        api_base=OLLAMA_API_BASE,
        api_key="",
        max_tokens=MAX_TOKENS
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
