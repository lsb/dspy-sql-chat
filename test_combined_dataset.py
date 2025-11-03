"""Test the combined dataset loading functionality."""
from dataset_loader import load_dataset


def test_dataset_loading():
    """Test that datasets load correctly and have expected sizes."""
    print("=" * 80)
    print("Testing Combined Dataset Loading")
    print("=" * 80)

    # Load all three datasets
    print("\n1. Loading datasets...")
    legitimate = load_dataset("legitimate.jsonl")
    policy_violations = load_dataset("content_policy_violation.jsonl")
    readonly_violations = load_dataset("read_only_violation.jsonl")

    print(f"   Legitimate examples: {len(legitimate)}")
    print(f"   Content policy violations: {len(policy_violations)}")
    print(f"   Read-only violations: {len(readonly_violations)}")

    # Verify expected sizes
    assert len(legitimate) == 100, f"Expected 100 legitimate examples, got {len(legitimate)}"
    assert len(policy_violations) == 20, f"Expected 20 policy violations, got {len(policy_violations)}"
    assert len(readonly_violations) == 20, f"Expected 20 read-only violations, got {len(readonly_violations)}"
    print("   ✓ All datasets loaded with expected sizes")

    # Test development mode split (5+5+5 train, 5+5+5 val)
    print("\n2. Testing development mode split...")
    train_dev = legitimate[:5] + policy_violations[:5] + readonly_violations[:5]
    val_dev = legitimate[5:10] + policy_violations[5:10] + readonly_violations[5:10]

    print(f"   Training examples: {len(train_dev)} (expected 15)")
    print(f"   Validation examples: {len(val_dev)} (expected 15)")
    assert len(train_dev) == 15, f"Expected 15 training examples, got {len(train_dev)}"
    assert len(val_dev) == 15, f"Expected 15 validation examples, got {len(val_dev)}"
    print("   ✓ Development mode split correct")

    # Test full mode split (25% val, 75% train)
    print("\n3. Testing full mode split...")

    # Split legitimate
    leg_val_size = int(len(legitimate) * 0.25)
    leg_train = legitimate[leg_val_size:]
    leg_val = legitimate[:leg_val_size]
    print(f"   Legitimate: {len(leg_train)} train, {len(leg_val)} val")

    # Split policy violations
    pol_val_size = int(len(policy_violations) * 0.25)
    pol_train = policy_violations[pol_val_size:]
    pol_val = policy_violations[:pol_val_size]
    print(f"   Policy violations: {len(pol_train)} train, {len(pol_val)} val")

    # Split read-only violations
    ro_val_size = int(len(readonly_violations) * 0.25)
    ro_train = readonly_violations[ro_val_size:]
    ro_val = readonly_violations[:ro_val_size]
    print(f"   Read-only violations: {len(ro_train)} train, {len(ro_val)} val")

    # Calculate duplication
    total_violations_train = len(pol_train) + len(ro_train)
    duplication_factor = len(leg_train) // total_violations_train
    remainder = len(leg_train) % total_violations_train

    print(f"\n   Total training violations: {total_violations_train}")
    print(f"   Legitimate training examples: {len(leg_train)}")
    print(f"   Duplication factor: {duplication_factor}")
    print(f"   Remainder: {remainder}")

    # Duplicate violation examples
    duplicated_violations = []
    for _ in range(duplication_factor):
        duplicated_violations.extend(pol_train)
        duplicated_violations.extend(ro_train)
    if remainder > 0:
        all_violations = pol_train + ro_train
        duplicated_violations.extend(all_violations[:remainder])

    train_full = leg_train + duplicated_violations
    val_full = leg_val + pol_val + ro_val

    print(f"\n   Final training set: {len(train_full)} examples")
    print(f"   Final validation set: {len(val_full)} examples")

    # Verify the training set has balanced legitimate and violations
    expected_train_size = len(leg_train) * 2  # Should be roughly 2x legitimate count
    assert len(train_full) == expected_train_size, f"Expected ~{expected_train_size} training examples, got {len(train_full)}"
    print("   ✓ Full mode split correct")

    # Test data structure
    print("\n4. Testing data structure...")
    for dataset_name, dataset in [("legitimate", legitimate[:1]),
                                   ("policy_violations", policy_violations[:1]),
                                   ("readonly_violations", readonly_violations[:1])]:
        example = dataset[0]
        assert 'question' in example, f"{dataset_name} missing 'question' field"
        assert 'sql' in example, f"{dataset_name} missing 'sql' field"
        print(f"   ✓ {dataset_name} has correct structure")

    # Test violation responses
    print("\n5. Testing violation responses...")

    # Check policy violations return correct message
    for pv in policy_violations:
        assert pv['sql'] == "SELECT 'query violates content policy'", \
            f"Policy violation has wrong SQL: {pv['sql']}"
    print("   ✓ All policy violations return correct message")

    # Check read-only violations return correct message
    for rv in readonly_violations:
        assert rv['sql'] == "SELECT 'database is read-only'", \
            f"Read-only violation has wrong SQL: {rv['sql']}"
    print("   ✓ All read-only violations return correct message")

    # Show sample examples from each category
    print("\n6. Sample examples:")
    print("\n   Legitimate:")
    print(f"   Q: {legitimate[0]['question']}")
    print(f"   SQL: {legitimate[0]['sql'][:80]}...")

    print("\n   Content Policy Violation:")
    print(f"   Q: {policy_violations[0]['question']}")
    print(f"   SQL: {policy_violations[0]['sql']}")

    print("\n   Read-Only Violation:")
    print(f"   Q: {readonly_violations[0]['question']}")
    print(f"   SQL: {readonly_violations[0]['sql']}")

    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)


if __name__ == "__main__":
    test_dataset_loading()
