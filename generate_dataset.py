"""Generate question-SQL pairs for training/evaluation."""
import json
import random
from db import create_db

# Question templates with corresponding SQL templates
QUESTION_TEMPLATES = [
    # Count queries
    {
        "question": "How many papers were published in {year}?",
        "sql": "SELECT COUNT(*) FROM paper_authorships WHERE Year = {year}"
    },
    {
        "question": "How many papers are there from {conference}?",
        "sql": "SELECT COUNT(*) FROM paper_authorships WHERE Conference = '{conference}'"
    },
    {
        "question": "How many distinct authors published in {year}?",
        "sql": "SELECT COUNT(DISTINCT Author) FROM paper_authorships WHERE Year = {year}"
    },
    {
        "question": "How many distinct affiliations are represented in {conference}?",
        "sql": "SELECT COUNT(DISTINCT Affiliation) FROM paper_authorships WHERE Conference = '{conference}'"
    },
    {
        "question": "How many papers did {author} publish?",
        "sql": "SELECT COUNT(*) FROM paper_authorships WHERE Author = '{author}'"
    },
    {
        "question": "How many total years of data do we have?",
        "sql": "SELECT COUNT(DISTINCT Year) FROM paper_authorships"
    },
    {
        "question": "How many different conferences are in the database?",
        "sql": "SELECT COUNT(DISTINCT Conference) FROM paper_authorships"
    },
    {
        "question": "How many papers were published between {year1} and {year2}?",
        "sql": "SELECT COUNT(*) FROM paper_authorships WHERE Year BETWEEN {year1} AND {year2}"
    },

    # List/selection queries
    {
        "question": "What are all the conferences in the database?",
        "sql": "SELECT DISTINCT Conference FROM paper_authorships ORDER BY Conference"
    },
    {
        "question": "What years are covered in the dataset?",
        "sql": "SELECT DISTINCT Year FROM paper_authorships ORDER BY Year"
    },
    {
        "question": "What papers were published in {year}?",
        "sql": "SELECT DISTINCT Title FROM paper_authorships WHERE Year = {year}"
    },
    {
        "question": "Who published papers at {conference} in {year}?",
        "sql": "SELECT DISTINCT Author FROM paper_authorships WHERE Conference = '{conference}' AND Year = {year}"
    },
    {
        "question": "What papers did {author} write?",
        "sql": "SELECT DISTINCT Title FROM paper_authorships WHERE Author = '{author}'"
    },
    {
        "question": "Which authors published at all three conferences?",
        "sql": "SELECT Author FROM paper_authorships GROUP BY Author HAVING COUNT(DISTINCT Conference) = 3"
    },

    # Aggregation queries
    {
        "question": "What is the earliest year in the database?",
        "sql": "SELECT MIN(Year) FROM paper_authorships"
    },
    {
        "question": "What is the most recent year in the database?",
        "sql": "SELECT MAX(Year) FROM paper_authorships"
    },
    {
        "question": "How many papers per year are there?",
        "sql": "SELECT Year, COUNT(*) as paper_count FROM paper_authorships GROUP BY Year ORDER BY Year"
    },
    {
        "question": "How many papers per conference?",
        "sql": "SELECT Conference, COUNT(*) as paper_count FROM paper_authorships GROUP BY Conference"
    },
    {
        "question": "Which year had the most papers?",
        "sql": "SELECT Year, COUNT(*) as paper_count FROM paper_authorships GROUP BY Year ORDER BY paper_count DESC LIMIT 1"
    },
    {
        "question": "Which conference has the most papers?",
        "sql": "SELECT Conference, COUNT(*) as paper_count FROM paper_authorships GROUP BY Conference ORDER BY paper_count DESC LIMIT 1"
    },
    {
        "question": "Who are the top 10 most prolific authors?",
        "sql": "SELECT Author, COUNT(*) as paper_count FROM paper_authorships GROUP BY Author ORDER BY paper_count DESC LIMIT 10"
    },
    {
        "question": "Which affiliation has the most papers?",
        "sql": "SELECT Affiliation, COUNT(*) as paper_count FROM paper_authorships GROUP BY Affiliation ORDER BY paper_count DESC LIMIT 1"
    },

    # Filter queries
    {
        "question": "Show papers from {affiliation}",
        "sql": "SELECT DISTINCT Title FROM paper_authorships WHERE Affiliation = '{affiliation}'"
    },
    {
        "question": "Which authors are affiliated with {affiliation}?",
        "sql": "SELECT DISTINCT Author FROM paper_authorships WHERE Affiliation = '{affiliation}'"
    },
    {
        "question": "What papers were published at {conference} after {year}?",
        "sql": "SELECT DISTINCT Title FROM paper_authorships WHERE Conference = '{conference}' AND Year > {year}"
    },
    {
        "question": "How many authors from {affiliation} published at {conference}?",
        "sql": "SELECT COUNT(DISTINCT Author) FROM paper_authorships WHERE Affiliation = '{affiliation}' AND Conference = '{conference}'"
    },

    # Self-JOIN queries (finding co-authors)
    {
        "question": "Who were the co-authors of {author}?",
        "sql": "SELECT DISTINCT p2.Author FROM paper_authorships p1 JOIN paper_authorships p2 ON p1.Title = p2.Title WHERE p1.Author = '{author}' AND p2.Author != '{author}'"
    },
    {
        "question": "Which authors collaborated with {author} at {conference}?",
        "sql": "SELECT DISTINCT p2.Author FROM paper_authorships p1 JOIN paper_authorships p2 ON p1.Title = p2.Title AND p1.Conference = p2.Conference WHERE p1.Author = '{author}' AND p2.Author != '{author}' AND p1.Conference = '{conference}'"
    },
    {
        "question": "How many co-authors did {author} have?",
        "sql": "SELECT COUNT(DISTINCT p2.Author) FROM paper_authorships p1 JOIN paper_authorships p2 ON p1.Title = p2.Title WHERE p1.Author = '{author}' AND p2.Author != '{author}'"
    },
    {
        "question": "Which authors from {affiliation} published together?",
        "sql": "SELECT DISTINCT p1.Author, p2.Author FROM paper_authorships p1 JOIN paper_authorships p2 ON p1.Title = p2.Title WHERE p1.Affiliation = '{affiliation}' AND p2.Affiliation = '{affiliation}' AND p1.Author < p2.Author"
    },
    {
        "question": "What papers have authors from multiple affiliations?",
        "sql": "SELECT DISTINCT p1.Title FROM paper_authorships p1 JOIN paper_authorships p2 ON p1.Title = p2.Title WHERE p1.Affiliation != p2.Affiliation"
    },

    # Subquery queries
    {
        "question": "Which authors published more than the average number of papers?",
        "sql": "SELECT Author, COUNT(*) as paper_count FROM paper_authorships GROUP BY Author HAVING COUNT(*) > (SELECT AVG(cnt) FROM (SELECT COUNT(*) as cnt FROM paper_authorships GROUP BY Author))"
    },
    {
        "question": "What papers were published in the year with the most publications?",
        "sql": "SELECT DISTINCT Title FROM paper_authorships WHERE Year = (SELECT Year FROM paper_authorships GROUP BY Year ORDER BY COUNT(*) DESC LIMIT 1)"
    },
    {
        "question": "Which affiliations had more papers than {affiliation}?",
        "sql": "SELECT Affiliation, COUNT(*) as cnt FROM paper_authorships GROUP BY Affiliation HAVING COUNT(*) > (SELECT COUNT(*) FROM paper_authorships WHERE Affiliation = '{affiliation}')"
    },
    {
        "question": "Who are the authors who published every year from {year1} to {year2}?",
        "sql": "SELECT Author FROM paper_authorships WHERE Year BETWEEN {year1} AND {year2} GROUP BY Author HAVING COUNT(DISTINCT Year) = {year2} - {year1} + 1"
    },

    # HAVING clause queries (more complex aggregations)
    {
        "question": "Which authors published more than 5 papers?",
        "sql": "SELECT Author, COUNT(*) as paper_count FROM paper_authorships GROUP BY Author HAVING COUNT(*) > 5 ORDER BY paper_count DESC"
    },
    {
        "question": "Which affiliations have published at all three conferences?",
        "sql": "SELECT Affiliation FROM paper_authorships GROUP BY Affiliation HAVING COUNT(DISTINCT Conference) = 3"
    },
    {
        "question": "Which authors published at {conference} in multiple years?",
        "sql": "SELECT Author FROM paper_authorships WHERE Conference = '{conference}' GROUP BY Author HAVING COUNT(DISTINCT Year) > 1"
    },
    {
        "question": "Which papers have more than 3 authors?",
        "sql": "SELECT Title, COUNT(DISTINCT Author) as author_count FROM paper_authorships GROUP BY Title HAVING COUNT(DISTINCT Author) > 3"
    },
    {
        "question": "Which years had more than 5000 papers published?",
        "sql": "SELECT Year, COUNT(*) as paper_count FROM paper_authorships GROUP BY Year HAVING COUNT(*) > 5000"
    },

    # Complex WHERE with multiple conditions
    {
        "question": "Which authors from {affiliation} published at {conference} after {year}?",
        "sql": "SELECT DISTINCT Author FROM paper_authorships WHERE Affiliation = '{affiliation}' AND Conference = '{conference}' AND Year > {year}"
    },
    {
        "question": "How many papers were published at {conference} between {year1} and {year2} by authors from {affiliation}?",
        "sql": "SELECT COUNT(DISTINCT Title) FROM paper_authorships WHERE Conference = '{conference}' AND Year BETWEEN {year1} AND {year2} AND Affiliation = '{affiliation}'"
    },

    # Set operations (INTERSECT/EXCEPT via subqueries)
    {
        "question": "Which authors published at {conference} but not at ICML?",
        "sql": "SELECT DISTINCT Author FROM paper_authorships WHERE Conference = '{conference}' AND Author NOT IN (SELECT Author FROM paper_authorships WHERE Conference = 'ICML')"
    },
    {
        "question": "Which authors published in both {year1} and {year2}?",
        "sql": "SELECT DISTINCT p1.Author FROM paper_authorships p1 JOIN paper_authorships p2 ON p1.Author = p2.Author WHERE p1.Year = {year1} AND p2.Year = {year2}"
    },
]


def get_sample_data():
    """Get sample data for template filling."""
    conn = create_db()
    cursor = conn.cursor()

    # Get sample years
    cursor.execute("SELECT DISTINCT Year FROM paper_authorships ORDER BY Year")
    years = [row[0] for row in cursor.fetchall()]

    # Get conferences
    cursor.execute("SELECT DISTINCT Conference FROM paper_authorships")
    conferences = [row[0] for row in cursor.fetchall()]

    # Get sample authors (from those with multiple papers)
    cursor.execute("""
        SELECT Author, COUNT(*) as cnt
        FROM paper_authorships
        GROUP BY Author
        HAVING cnt >= 3
        ORDER BY RANDOM()
        LIMIT 20
    """)
    authors = [row[0] for row in cursor.fetchall()]

    # Get sample affiliations (from those with multiple papers)
    cursor.execute("""
        SELECT Affiliation, COUNT(*) as cnt
        FROM paper_authorships
        GROUP BY Affiliation
        HAVING cnt >= 10
        ORDER BY RANDOM()
        LIMIT 20
    """)
    affiliations = [row[0] for row in cursor.fetchall()]

    conn.close()

    return {
        'years': years,
        'conferences': conferences,
        'authors': authors,
        'affiliations': affiliations
    }


def generate_question_sql_pair(template, sample_data):
    """Generate a question-SQL pair from a template."""
    question = template['question']
    sql = template['sql']

    # Replace placeholders
    if '{year}' in question:
        year = random.choice(sample_data['years'])
        question = question.replace('{year}', str(year))
        sql = sql.replace('{year}', str(year))

    if '{year1}' in question:
        year1 = random.choice(sample_data['years'][:-1])
        year2 = random.choice([y for y in sample_data['years'] if y > year1])
        question = question.replace('{year1}', str(year1)).replace('{year2}', str(year2))
        sql = sql.replace('{year1}', str(year1)).replace('{year2}', str(year2))

    if '{conference}' in question:
        conference = random.choice(sample_data['conferences'])
        question = question.replace('{conference}', conference)
        sql = sql.replace('{conference}', conference)

    if '{author}' in question:
        author = random.choice(sample_data['authors'])
        question = question.replace('{author}', author)
        # Escape single quotes in SQL
        sql = sql.replace('{author}', author.replace("'", "''"))

    if '{affiliation}' in question:
        affiliation = random.choice(sample_data['affiliations'])
        question = question.replace('{affiliation}', affiliation)
        # Escape single quotes in SQL
        sql = sql.replace('{affiliation}', affiliation.replace("'", "''"))

    return {
        'question': question,
        'sql': sql
    }


def validate_sql(sql):
    """Validate that SQL executes without error."""
    try:
        conn = create_db()
        cursor = conn.cursor()
        cursor.execute(sql)
        cursor.fetchall()
        conn.close()
        return True
    except Exception as e:
        print(f"Invalid SQL: {sql}")
        print(f"Error: {e}")
        return False


def generate_dataset(num_pairs=200, output_file="question_sql_pairs.jsonl"):
    """Generate dataset of question-SQL pairs."""
    sample_data = get_sample_data()
    pairs = []

    # Generate pairs ensuring diversity
    attempts = 0
    max_attempts = num_pairs * 3

    while len(pairs) < num_pairs and attempts < max_attempts:
        attempts += 1

        # Pick a random template
        template = random.choice(QUESTION_TEMPLATES)

        # Generate pair
        pair = generate_question_sql_pair(template, sample_data)

        # Validate SQL
        if not validate_sql(pair['sql']):
            continue

        # Avoid exact duplicates
        if pair not in pairs:
            pairs.append(pair)

    # Write to JSONL file
    with open(output_file, 'w') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')

    print(f"Generated {len(pairs)} question-SQL pairs")
    print(f"Saved to {output_file}")

    return pairs


if __name__ == "__main__":
    pairs = generate_dataset(200)

    # Show some examples
    print("\n=== Sample pairs ===")
    for pair in random.sample(pairs, min(5, len(pairs))):
        print(f"\nQ: {pair['question']}")
        print(f"SQL: {pair['sql']}")
