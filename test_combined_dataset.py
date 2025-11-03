"""Test the combined dataset loading functionality."""
from dataset_loader import load_dataset_as_dspy_examples, load_combined_dataset


def test_dataset_loading():
    """Test that datasets load correctly and have expected sizes."""
    print("=" * 80)
    print("Testing Combined Dataset Loading")
    print("=" * 80)

    # Load individual datasets as DSPy examples
    print("\n1. Loading individual datasets as DSPy examples...")
    legitimate = load_dataset_as_dspy_examples("legitimate.jsonl")
    policy_violations = load_dataset_as_dspy_examples("content_policy_violation.jsonl")
    readonly_violations = load_dataset_as_dspy_examples("read_only_violation.jsonl")

    print(f"   Legitimate examples: {len(legitimate)}")
    print(f"   Content policy violations: {len(policy_violations)}")
    print(f"   Read-only violations: {len(readonly_violations)}")

    # Verify expected sizes
    assert len(legitimate) == 100, f"Expected 100 legitimate examples, got {len(legitimate)}"
    assert len(policy_violations) == 20, f"Expected 20 policy violations, got {len(policy_violations)}"
    assert len(readonly_violations) == 20, f"Expected 20 read-only violations, got {len(readonly_violations)}"
    print("   ✓ All datasets loaded with expected sizes")

    # Test DSPy Example structure
    print("\n2. Testing DSPy Example structure...")
    example = legitimate[0]
    assert hasattr(example, 'natural_language_query'), "Missing natural_language_query field"
    assert hasattr(example, 'sql_query'), "Missing sql_query field"
    print(f"   Example query: {example.natural_language_query[:60]}...")
    print(f"   Example SQL: {example.sql_query[:60]}...")
    print("   ✓ DSPy Examples have correct structure")

    # Test development mode
    print("\n3. Testing development mode (5+5+5 train, 5+5+5 val)...")
    train_dev, val_dev = load_combined_dataset(development_mode=True)
    assert len(train_dev) == 15, f"Expected 15 training examples, got {len(train_dev)}"
    assert len(val_dev) == 15, f"Expected 15 validation examples, got {len(val_dev)}"
    print("   ✓ Development mode split correct")

    # Test full mode
    print("\n4. Testing full mode (25% val, duplicate violations)...")
    train_full, val_full = load_combined_dataset(development_mode=False)

    # Expected counts:
    # - Legitimate: 75 train, 25 val
    # - Policy violations: 15 train, 5 val
    # - Read-only violations: 15 train, 5 val
    # - Training violations duplicated to match legitimate: 75
    # - Total train: 75 + 75 = 150
    # - Total val: 25 + 5 + 5 = 35
    assert len(train_full) == 150, f"Expected 150 training examples, got {len(train_full)}"
    assert len(val_full) == 35, f"Expected 35 validation examples, got {len(val_full)}"
    print("   ✓ Full mode split correct")

    # Test violation SQL responses
    print("\n5. Testing violation responses...")
    policy_example = policy_violations[0]
    readonly_example = readonly_violations[0]

    assert policy_example.sql_query == "SELECT 'query violates content policy'", \
        f"Policy violation has wrong SQL: {policy_example.sql_query}"
    assert readonly_example.sql_query == "SELECT 'database is read-only'", \
        f"Read-only violation has wrong SQL: {readonly_example.sql_query}"
    print("   ✓ All violation responses correct")

    # Show sample examples
    print("\n6. Sample examples:")
    print("\n   Legitimate:")
    print(f"   Q: {legitimate[0].natural_language_query}")
    print(f"   SQL: {legitimate[0].sql_query[:80]}...")

    print("\n   Content Policy Violation:")
    print(f"   Q: {policy_violations[0].natural_language_query}")
    print(f"   SQL: {policy_violations[0].sql_query}")

    print("\n   Read-Only Violation:")
    print(f"   Q: {readonly_violations[0].natural_language_query}")
    print(f"   SQL: {readonly_violations[0].sql_query}")

    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)


if __name__ == "__main__":
    test_dataset_loading()
