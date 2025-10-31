from db import create_db


def main():
    print("Hello from dspy-sql-chat!")

    # Create database and run a sample query
    conn = create_db()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(DISTINCT year) FROM paper_authorships")
    year_count = cursor.fetchone()[0]
    print(f"Database loaded successfully! Found {year_count} distinct years.")

    conn.close()


if __name__ == "__main__":
    main()
