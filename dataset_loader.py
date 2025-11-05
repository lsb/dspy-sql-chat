"""Dataset loading utilities for SQL chat training and evaluation.

This module provides functions to load and combine datasets from three sources:
- legitimate.jsonl: Valid SQL queries
- content_policy_violation.jsonl: Queries that violate content policy
- read_only_violation.jsonl: Queries that attempt to modify the database
"""

import json
import random
import dspy
from db import create_db
from query_timeout import (
    measure_baseline_query_time,
    execute_query_with_timeout,
    get_query_timeout
)


def load_dataset_as_dspy_examples(filepath):
    """
    Load a JSONL dataset file as DSPy Examples.

    Args:
        filepath: Path to the JSONL file

    Returns:
        List of dspy.Example objects with natural_language_query and sql_query fields
    """
    examples = []
    with open(filepath, 'r') as f:
        for line in f:
            pair = json.loads(line)
            # Create DSPy Example with input and output fields
            example = dspy.Example(
                natural_language_query=pair['question'],
                sql_query=pair['sql']
            ).with_inputs('natural_language_query')
            examples.append(example)
    return examples


def filter_slow_queries(examples, timeout_seconds, csv_path="papers.csv"):
    """
    Filter out examples with SQL queries that exceed the timeout.

    Args:
        examples: List of dspy.Example objects with sql_query field
        timeout_seconds: Maximum time allowed for query execution
        csv_path: Path to the CSV file for database creation

    Returns:
        List of examples that complete within the timeout
    """
    filtered_examples = []
    skipped_count = 0

    conn = create_db(csv_path)

    for i, example in enumerate(examples):
        success, results, error = execute_query_with_timeout(
            conn,
            example.sql_query,
            timeout_seconds=timeout_seconds
        )

        if success:
            filtered_examples.append(example)
        else:
            skipped_count += 1
            print(f"   Skipping query {i+1}/{len(examples)}: {error}")
            print(f"      Query: {example.sql_query[:100]}...")

    conn.close()

    if skipped_count > 0:
        print(f"   Total skipped: {skipped_count}/{len(examples)} queries")

    return filtered_examples


def load_combined_dataset(development_mode=False, random_seed=42):
    """
    Load and combine datasets from three sources.

    In development mode:
        - 5 legitimate, 5 policy violations, 5 read-only violations for training (15 total)
        - 5 legitimate, 5 policy violations, 5 read-only violations for validation (15 total)

    In full mode:
        - Use 25% of each dataset for validation, 75% for training
        - Duplicate training violations to match the number of legitimate training examples

    Legitimate queries are filtered to exclude queries that take longer than 50x
    the baseline query (SELECT COUNT(DISTINCT year) FROM paper_authorships).

    The training set is shuffled using a fixed random seed for reproducibility.

    Args:
        development_mode: If True, use small subset for development
        random_seed: Seed for shuffling training set (default: 42)

    Returns:
        Tuple of (train_examples, val_examples) as dspy.Example objects
    """
    # Measure baseline query time for timeout calculation
    print("Measuring baseline query time...")
    baseline_time = measure_baseline_query_time()
    timeout = get_query_timeout(baseline_time, multiplier=50)
    print(f"   Baseline time: {baseline_time:.4f}s, Timeout: {timeout:.4f}s (50x baseline)")

    # Load all three datasets
    print("Loading datasets...")
    legitimate = load_dataset_as_dspy_examples("legitimate.jsonl")
    policy_violations = load_dataset_as_dspy_examples("content_policy_violation.jsonl")
    readonly_violations = load_dataset_as_dspy_examples("read_only_violation.jsonl")

    # Filter legitimate queries that exceed timeout
    print(f"Filtering legitimate queries (loaded {len(legitimate)} queries)...")
    legitimate = filter_slow_queries(legitimate, timeout)
    print(f"   Kept {len(legitimate)} legitimate queries after filtering")

    if development_mode:
        # Development mode: 5 from each for training, 5 from each for validation
        train_examples = legitimate[:5] + policy_violations[:5] + readonly_violations[:5]
        val_examples = legitimate[5:10] + policy_violations[5:10] + readonly_violations[5:10]
    else:
        # Full mode: 25% validation, 75% training
        # Split legitimate
        leg_val_size = int(len(legitimate) * 0.25)
        leg_train = legitimate[leg_val_size:]
        leg_val = legitimate[:leg_val_size]

        # Split policy violations
        pol_val_size = int(len(policy_violations) * 0.25)
        pol_train = policy_violations[pol_val_size:]
        pol_val = policy_violations[:pol_val_size]

        # Split read-only violations
        ro_val_size = int(len(readonly_violations) * 0.25)
        ro_train = readonly_violations[ro_val_size:]
        ro_val = readonly_violations[:ro_val_size]

        # Duplicate violations to match legitimate count
        all_violations = pol_train + ro_train
        duplication_factor = len(leg_train) // len(all_violations)
        remainder = len(leg_train) % len(all_violations)
        duplicated_violations = (all_violations * duplication_factor) + all_violations[:remainder]

        # Combine training and validation sets
        train_examples = leg_train + duplicated_violations
        val_examples = leg_val + pol_val + ro_val

    # Shuffle training set with fixed seed for reproducibility
    random.seed(random_seed)
    random.shuffle(train_examples)
    print(f"   Shuffled training set with seed: {random_seed}")

    print(f"   Training examples: {len(train_examples)}")
    print(f"   Validation examples: {len(val_examples)}")

    return train_examples, val_examples


if __name__ == "__main__":
    # Test the dataset loading
    print("Testing dataset loading...")

    print("\nDevelopment mode:")
    train, val = load_combined_dataset(development_mode=True)
    print(f"Train: {len(train)}, Val: {len(val)}")

    print("\nFull mode:")
    train, val = load_combined_dataset(development_mode=False)
    print(f"Train: {len(train)}, Val: {len(val)}")
