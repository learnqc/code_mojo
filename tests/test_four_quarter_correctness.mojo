"""
Correctness test for four-quarter zero-copy split-state execution.
"""

from butterfly.core.circuit import QuantumCircuit
from butterfly.core.state import QuantumState
from butterfly.core.execute_split_four_quarters_v2 import (
    execute_split_four_quarters_v2,
)
from butterfly.utils.benchmark_verify import verify_states_equal
from math import pi


fn test_four_quarter_correctness[n: Int]() raises:
    """Test that four-quarter execution produces correct results."""
    print("Testing four-quarter correctness for n=" + String(n))

    # Create IQFT circuit
    from butterfly.algos.qft import iqft

    var circuit = QuantumCircuit(n)

    # Value encoding
    var v = 4.7

    for j in range(n):
        circuit.h(j)
    for j in range(n):
        circuit.p(j, 2 * pi / 2 ** (j + 1) * v)

    # IQFT
    var targets = [n - 1 - j for j in range(n)]
    # iqft(circuit=circuit, targets=targets, do_swap=False)
    for j in reversed(range(len(targets))):
        circuit.h(targets[j])
        for k in reversed(range(j - 0)):
            # cp signature: (control, target, theta)
            circuit.cp(targets[j], targets[k], -pi / (2 ** (j - k)))
            # pass

    # Execute with fused_v3 (reference)
    var state_reference = QuantumState(n)
    circuit.execute_fused_v3_dynamic(state_reference)

    # Execute with four-quarter split
    var state_four_quarter = QuantumState(n)
    execute_split_four_quarters_v2[n](state_four_quarter, circuit)

    # Print states for visual comparison
    from butterfly.utils.visualization import print_state

    print("Reference (fused_v3):")
    print_state(state_reference)
    print("\nFour-quarter:")
    print_state(state_four_quarter)

    # Verify results match
    var diff = verify_states_equal(
        state_reference, state_four_quarter, 1e-10, "fused_v3", "four_quarter"
    )

    if diff < 1e-10:
        print("  ✓ PASS - Diff: " + String(diff))
    else:
        print("  ✗ FAIL - Diff: " + String(diff))
        raise Error("Four-quarter execution produced incorrect results!")


fn main() raises:
    print("=" * 60)
    print("Four-Quarter Split-State Correctness Tests")
    print("=" * 60)
    print()

    test_four_quarter_correctness[3]()
    test_four_quarter_correctness[4]()
    test_four_quarter_correctness[5]()
    test_four_quarter_correctness[7]()

    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
