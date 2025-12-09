from butterfly.core.state import QuantumState, generate_state
from butterfly.core.types import *
from testing import assert_true, assert_almost_equal
from math import sqrt


def test_iterator_basic():
    # Test iterating over a simple state
    var state = QuantumState(2)
    state[0] = Amplitude(FloatType(1.0 / sqrt(2.0)), FloatType(0.0))
    state[1] = Amplitude(FloatType(0.0), FloatType(0.0))
    state[2] = Amplitude(FloatType(0.0), FloatType(0.0))
    state[3] = Amplitude(FloatType(1.0 / sqrt(2.0)), FloatType(0.0))

    var count = 0
    for amp in state:
        count += 1

    assert_true(count == 4)
    print("✓ Iterator basic test passed")


def test_iterator_values():
    # Test that iterator returns correct values
    var state = QuantumState(2)
    state[0] = Amplitude(FloatType(0.5), FloatType(0.0))
    state[1] = Amplitude(FloatType(0.0), FloatType(0.5))
    state[2] = Amplitude(FloatType(0.5), FloatType(0.0))
    state[3] = Amplitude(FloatType(0.0), FloatType(0.5))

    var expected_re = List[FloatType](0.5, 0.0, 0.5, 0.0)
    var expected_im = List[FloatType](0.0, 0.5, 0.0, 0.5)

    var idx = 0
    for amp in state:
        assert_almost_equal(Float64(amp.re), Float64(expected_re[idx]), atol=1e-10)
        assert_almost_equal(Float64(amp.im), Float64(expected_im[idx]), atol=1e-10)
        idx += 1

    print("✓ Iterator values test passed")


def test_iterator_random_state():
    # Test iterating over a randomly generated state
    var state = generate_state(3, seed=42)

    var count = 0
    var total_prob = FloatType(0.0)

    for amp in state:
        total_prob += amp.re * amp.re + amp.im * amp.im
        count += 1

    assert_true(count == 8)
    assert_almost_equal(Float64(total_prob), 1.0, atol=1e-6)

    print("✓ Iterator random state test passed")


def test_iterator_matches_indexing():
    # Test that iterator yields same values as indexing
    var state = generate_state(3, seed=99)

    var idx = 0
    for amp in state:
        var indexed_amp = state[idx]
        assert_almost_equal(Float64(amp.re), Float64(indexed_amp.re), atol=1e-10)
        assert_almost_equal(Float64(amp.im), Float64(indexed_amp.im), atol=1e-10)
        idx += 1

    print("✓ Iterator matches indexing test passed")


def test_multiple_iterations():
    # Test that we can iterate multiple times
    var state = QuantumState(2)
    state[0] = Amplitude(FloatType(1.0), FloatType(0.0))
    state[1] = Amplitude(FloatType(0.0), FloatType(1.0))
    state[2] = Amplitude(FloatType(0.5), FloatType(0.5))
    state[3] = Amplitude(FloatType(0.0), FloatType(0.0))

    var count1 = 0
    for amp in state:
        count1 += 1

    var count2 = 0
    for amp in state:
        count2 += 1

    assert_true(count1 == 4)
    assert_true(count2 == 4)

    print("✓ Multiple iterations test passed")


def main():
    print("Running QuantumState iterator tests...\n")

    test_iterator_basic()
    test_iterator_values()
    test_iterator_random_state()
    test_iterator_matches_indexing()
    test_multiple_iterations()

    print("\n✅ All iterator tests passed!")
