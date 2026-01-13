"""
Reusable verification utilities for benchmarks.

Provides functions to verify correctness before benchmarking,
serving as both validation and warmup.
"""

from butterfly.core.state import QuantumState
from butterfly.core.circuit import QuantumCircuit
from butterfly.utils.visualization import print_state
from collections import List


fn verify_states_equal(
    state1: QuantumState,
    state2: QuantumState,
    tolerance: Float64 = 1e-5,
    name1: String = "state1",
    name2: String = "state2",
) raises -> Float64:
    """
    Verify two quantum states are approximately equal.

    Args:
        state1: First quantum state
        state2: Second quantum state
        tolerance: Maximum allowed difference
        name1: Name for first state (for error messages)
        name2: Name for second state (for error messages)

    Returns:
        Sum of squared differences

    Raises:
        Error if states differ by more than tolerance
    """
    if state1.size() != state2.size():
        raise Error(
            "State sizes differ: "
            + String(state1.size())
            + " vs "
            + String(state2.size())
        )

    var diff_sum = 0.0
    for idx in range(state1.size()):
        var diff_re = state1.re[idx] - state2.re[idx]
        var diff_im = state1.im[idx] - state2.im[idx]
        diff_sum += diff_re * diff_re + diff_im * diff_im

    if diff_sum > tolerance:
        print_state(state1)
        print_state(state2)
        raise Error(
            "States differ! "
            + name1
            + " vs "
            + name2
            + " - Diff: "
            + String(diff_sum)
            + " (tolerance: "
            + String(tolerance)
            + ")"
        )

    return diff_sum


fn verify_and_warmup[
    func1: fn () -> QuantumState,
    func2: fn () -> QuantumState,
](
    name1: String,
    name2: String,
    tolerance: Float64 = 1e-5,
    verbose: Bool = True,
) raises -> Float64:
    """
    Verify two functions produce the same result and warm up the code paths.

    Args:
        func1: First function to test
        func2: Second function to test
        name1: Name for first function
        name2: Name for second function
        tolerance: Maximum allowed difference
        verbose: Print verification status

    Returns:
        Sum of squared differences
    """
    if verbose:
        print("  Verifying " + name1 + " vs " + name2 + "...", end="")

    # Run both functions (serves as warmup)
    var state1 = func1()
    var state2 = func2()

    # Verify they match
    var diff = verify_states_equal(state1, state2, tolerance, name1, name2)

    if verbose:
        print(" ✓ (diff: " + String(diff) + ")")

    return diff
