"""Dataset loading utilities for SQL chat training and evaluation.

This module provides functions to load and combine datasets from three sources:
- legitimate.jsonl: Valid SQL queries
- content_policy_violation.jsonl: Queries that violate content policy
- read_only_violation.jsonl: Queries that attempt to modify the database
"""

import json
import random
import dspy
from itertools import cycle
from db import create_db
from query_timeout import execute_query_with_timeout


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


def filter_slow_queries(examples, timeout_seconds=10.0, csv_path="papers.csv"):
    """Filter out examples with SQL queries that exceed the timeout.

    Args:
        examples: List of dspy.Example objects with sql_query field
        timeout_seconds: Maximum time allowed for query execution (default: 10.0)
        csv_path: Path to the CSV file for database creation

    Returns:
        List of examples that complete within the timeout
    """
    filtered_examples = []
    skipped_count = 0

    conn = create_db(csv_path)

    for i, example in enumerate(examples):
        results, error = execute_query_with_timeout(
            conn,
            example.sql_query,
            timeout_seconds=timeout_seconds
        )

        if results is not None:
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
    """Load and combine datasets from three sources.

    In development mode:
        - 5 legitimate, 5 policy violations, 5 read-only violations for training (15 total)
        - 5 legitimate, 5 policy violations, 5 read-only violations for validation (15 total)

    In full mode:
        - Use 25% of each dataset for validation, 75% for training
        - Training interleaves: legitimate, content policy violation, read-only violation (repeating)

    Legitimate queries are filtered to exclude queries that take longer than 10 seconds.

    Args:
        development_mode: If True, use small subset for development
        random_seed: Seed for shuffling before splitting (default: 42)

    Returns:
        Tuple of (train_examples, val_examples) as dspy.Example objects
    """
    # Load all three datasets
    print("Loading datasets...")
    legitimate = load_dataset_as_dspy_examples("legitimate.jsonl")
    policy_violations = load_dataset_as_dspy_examples("content_policy_violation.jsonl")
    readonly_violations = load_dataset_as_dspy_examples("read_only_violation.jsonl")

    # Filter legitimate queries that exceed timeout
    print(f"Filtering legitimate queries (loaded {len(legitimate)} queries)...")
    legitimate = filter_slow_queries(legitimate)
    print(f"   Kept {len(legitimate)} legitimate queries after filtering")

    # Shuffle all datasets with fixed seed before splitting
    random.seed(random_seed)
    random.shuffle(legitimate)
    random.shuffle(policy_violations)
    random.shuffle(readonly_violations)

    if development_mode:
        # Development mode: 5 from each for training, 5 from each for validation
        leg_train, leg_val = legitimate[:5], legitimate[5:10]
        pol_train, pol_val = policy_violations[:5], policy_violations[5:10]
        ro_train, ro_val = readonly_violations[:5], readonly_violations[5:10]
    else:
        # Full mode: 25% validation, 75% training
        leg_val_size = int(len(legitimate) * 0.25)
        leg_train, leg_val = legitimate[leg_val_size:], legitimate[:leg_val_size]

        pol_val_size = int(len(policy_violations) * 0.25)
        pol_train, pol_val = policy_violations[pol_val_size:], policy_violations[:pol_val_size]

        ro_val_size = int(len(readonly_violations) * 0.25)
        ro_train, ro_val = readonly_violations[ro_val_size:], readonly_violations[:ro_val_size]

    # Interleave training examples: leg, pol, ro, leg, pol, ro, ...
    # Use cycle to repeat violations as needed to match legitimate count
    train_examples = []
    pol_cycle = cycle(pol_train)
    ro_cycle = cycle(ro_train)
    for leg_example in leg_train:
        train_examples.append(leg_example)
        train_examples.append(next(pol_cycle))
        train_examples.append(next(ro_cycle))

    # Combine validation sets (order doesn't matter as much for validation)
    val_examples = leg_val + pol_val + ro_val

    print(f"   Training examples: {len(train_examples)}")
    print(f"   Validation examples: {len(val_examples)}")

    return train_examples, val_examples
