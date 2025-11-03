"""Simple baseline evaluation using dspy.Evaluate."""
import json
import os
import dspy
from text_to_sql import TextToSQL, OLLAMA_MODEL, OLLAMA_API_BASE, MAX_TOKENS, clean_sql
from db import create_db
from sql_metric import sql_correctness_metric

# Check if running in development mode
DEVELOPMENT = os.environ.get('DEVELOPMENT', '0') == '1'
REFLECTION_LM = dspy.LM(
    "ollama_chat/gptoss20b-cpu",
    api_base=OLLAMA_API_BASE,
    api_key="",
    max_tokens=123456,
    timeout=86400,
    stream_timeout=86400,
)

def load_dataset(filepath="question_sql_pairs.jsonl"):
    """Load the question-SQL pairs dataset as DSPy Examples."""
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


def load_combined_dataset(development_mode=False):
    """
    Load and combine datasets from three sources.

    In development mode:
        - 5 legitimate, 5 policy violations, 5 read-only violations for training (15 total)
        - 5 legitimate, 5 policy violations, 5 read-only violations for validation (15 total)

    In full mode:
        - Use 25% of each dataset for validation, 75% for training
        - Duplicate training violations to match the number of legitimate training examples

    Args:
        development_mode: If True, use small subset for development

    Returns:
        Tuple of (train_examples, val_examples)
    """
    # Load all three datasets
    legitimate = load_dataset("legitimate.jsonl")
    policy_violations = load_dataset("content_policy_violation.jsonl")
    readonly_violations = load_dataset("read_only_violation.jsonl")

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

        # Calculate how many times to duplicate violations to match legitimate count
        total_violations_train = len(pol_train) + len(ro_train)
        if total_violations_train > 0:
            duplication_factor = len(leg_train) // total_violations_train
            remainder = len(leg_train) % total_violations_train
        else:
            duplication_factor = 0
            remainder = 0

        # Duplicate violation examples
        duplicated_violations = []
        for _ in range(duplication_factor):
            duplicated_violations.extend(pol_train)
            duplicated_violations.extend(ro_train)
        # Add partial violations to make up the remainder
        if remainder > 0:
            all_violations = pol_train + ro_train
            duplicated_violations.extend(all_violations[:remainder])

        # Combine training and validation sets
        train_examples = leg_train + duplicated_violations
        val_examples = leg_val + pol_val + ro_val

    print(f"   Training examples: {len(train_examples)}")
    print(f"   Validation examples: {len(val_examples)}")

    return train_examples, val_examples


def main():
    """Run baseline evaluation."""
    mode = "Development (15 train + 15 val)" if DEVELOPMENT else "Full Dataset"
    print("=" * 80)
    print(f"DSPy Text-to-SQL Baseline Evaluation - {mode}")
    print("=" * 80)

    # Load dataset
    print("\n1. Loading dataset...")
    train_examples, val_examples = load_combined_dataset(development_mode=DEVELOPMENT)
    all_examples = train_examples + val_examples
    print(f"   Loaded {len(all_examples)} total examples")

    # Set up language model
    print("\n2. Setting up language model...")
    task_model = dspy.LM(
        OLLAMA_MODEL,
        api_base=OLLAMA_API_BASE,
        api_key="",
        max_tokens=MAX_TOKENS,
        timeout=86400,
        stream_timeout=86400,
    )
    dspy.configure(lm=task_model)
    print(f"   Model: {task_model.model}")

    # Create predictor
    print("\n3. Creating predictor...")
    predictor = dspy.ChainOfThought(TextToSQL)

    # Use dspy.Evaluate for both modes (same code path)
    print("\n4. Evaluating baseline...")
    evaluator = dspy.Evaluate(
        devset=all_examples,
        metric=sql_correctness_metric,
        num_threads=1,  # Use 1 thread for stability
        display_progress=True,
        display_table=5  # Show first 5 results
    )

    result = evaluator(predictor)
    # Extract the score from the result (could be float or EvaluationResult)
    score = result if isinstance(result, (int, float)) else float(result)

    print("\n" + "=" * 80)
    print(f"Baseline Accuracy: {score:.1f}%")
    print("=" * 80)

    # Train and validation sets are already split by load_combined_dataset()
    print(f"\n5. Training with GEPA optimizer...")
    print(f"   Training set: {len(train_examples)} examples")
    print(f"   Validation set: {len(val_examples)} examples")

    optimizer = dspy.GEPA(
        metric=sql_correctness_metric,
        reflection_lm=REFLECTION_LM,
        track_stats=True,
        max_full_evals=1 if DEVELOPMENT else 5,
    )

    optimized_predictor = optimizer.compile(
        predictor,
        trainset=train_examples,
        valset=val_examples,
    )

    print(optimized_predictor)
    print(optimized_predictor.predict.signature.instructions)

    optimized_result = evaluator(optimized_predictor)
    optimized_score = optimized_result if isinstance(optimized_result, (int, float)) else float(optimized_result)

    print("\n" + "=" * 80)
    print(f"Optimized Baseline Accuracy: {optimized_score:.1f}%")
    print("=" * 80)

    # Manual examination of failures
    num_to_examine = len(all_examples) if DEVELOPMENT else 20
    print(f"\n6. Examining {'all' if DEVELOPMENT else 'first 20'} examples...")
    failures = []
    successes = []

    for example in all_examples[:num_to_examine]:
        pred = predictor(natural_language_query=example.natural_language_query)
        metric_result = sql_correctness_metric(example, pred)

        # Extract score from dspy.Prediction object
        score = metric_result.score if hasattr(metric_result, 'score') else float(metric_result)
        pred_sql = clean_sql(pred.sql_query) if hasattr(pred, 'sql_query') and pred.sql_query else None

        # Score >= 0.999 is considered correct
        if score >= 0.999:
            successes.append({
                'question': example.natural_language_query,
                'sql': example.sql_query,
                'pred_sql': pred_sql,
                'score': score
            })
        else:
            failures.append({
                'question': example.natural_language_query,
                'gold_sql': example.sql_query,
                'pred_sql': pred_sql,
                'score': score
            })

    print(f"\n   Successes: {len(successes)}/{num_to_examine}")
    print(f"   Failures: {len(failures)}/{num_to_examine}")

    if DEVELOPMENT:
        # In development mode, show ALL examples with detailed scores
        print("\n   All examples:")
        for i, failure in enumerate(failures, 1):
            status = "~" if failure['score'] >= 0.5 else "✗"
            print(f"\n   {i}. [{status}] Score: {failure['score']:.3f}")
            print(f"      Q: {failure['question']}")
            print(f"      Gold: {failure['gold_sql']}")
            print(f"      Pred: {failure['pred_sql']}")

        for i, success in enumerate(successes, len(failures) + 1):
            print(f"\n   {i}. [✓] Score: {success['score']:.3f}")
            print(f"      Q: {success['question']}")
            print(f"      SQL: {success['sql']}")
    else:
        # In full mode, show sample failures and successes
        print("\n   Sample failures:")
        for i, failure in enumerate(failures[:5], 1):
            print(f"\n   {i}. Q: {failure['question']}")
            print(f"      Gold: {failure['gold_sql']}")
            print(f"      Pred: {failure['pred_sql']}")

        print("\n   Sample successes:")
        for i, success in enumerate(successes[:3], 1):
            print(f"\n   {i}. Q: {success['question']}")
            print(f"      SQL: {success['sql']}")

    return score


if __name__ == "__main__":
    score = main()
