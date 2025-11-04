"""Simple baseline evaluation using dspy.Evaluate."""
import os
import dspy
from text_to_sql import TextToSQL, OLLAMA_BASE_MODEL, OLLAMA_API_BASE, MAX_TOKENS
from db import create_db
from sql_metric import sql_correctness_metric
from dataset_loader import load_combined_dataset

# Check if running in development mode
DEVELOPMENT = os.environ.get('DEVELOPMENT', '0') == '1'
REFLECTION_LM_MODEL = os.environ.get("REFLECTION_LM_MODEL", "ollama_chat/gpt-oss:20b")
REFLECTION_LM = dspy.LM(
    REFLECTION_LM_MODEL,
    api_base=OLLAMA_API_BASE,
    api_key="",
    max_tokens=123456,
    timeout=86400,
    stream_timeout=86400,
)


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
        OLLAMA_BASE_MODEL,
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
        pred_sql = pred.sql_query if hasattr(pred, 'sql_query') and pred.sql_query else None

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
