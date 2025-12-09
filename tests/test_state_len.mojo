from butterfly.core.state import QuantumState, generate_state
from testing import assert_true


def test_len_basic():
    var state = QuantumState(3)
    assert_true(len(state) == 8)
    print("✓ len() basic test passed")


def test_len_different_sizes():
    var state2 = QuantumState(2)
    assert_true(len(state2) == 4)

    var state4 = QuantumState(4)
    assert_true(len(state4) == 16)

    var state10 = QuantumState(10)
    assert_true(len(state10) == 1024)

    print("✓ len() different sizes test passed")


def test_len_generated_state():
    var state = generate_state(5, seed=42)
    assert_true(len(state) == 32)
    print("✓ len() generated state test passed")


def main():
    print("Running len() tests...\n")

    test_len_basic()
    test_len_different_sizes()
    test_len_generated_state()

    print("\n✅ All len() tests passed!")
