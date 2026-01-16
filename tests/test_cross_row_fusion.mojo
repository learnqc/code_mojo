"""Test cross-row HP fusion correctness."""
from butterfly.core.state import QuantumState
from butterfly.core.quantum_circuit import QuantumCircuit
from butterfly.core.executors import execute_scalar, execute_grid_fused
from butterfly.utils.benchmark_verify import verify_states_equal
from butterfly.utils.context import ExecContext
from math import pi


fn main() raises:
    print("Testing cross-row HP fusion...")
    var n = 12
    var col_bits = 6

    # Create circuit with cross-row H+P (both targets >= col_bits)
    var qc = QuantumCircuit(n)
    for i in range(3):
        qc.h(i)
    qc.h(col_bits)  # Cross-row H
    qc.p(col_bits + 1, pi / 4)  # Cross-row P (should fuse with H)

    var state_baseline = QuantumState(n)
    execute_scalar(state_baseline, qc)

    var state_fused = QuantumState(n)
    var ctx = ExecContext()
    execute_grid_fused(state_fused, qc, ctx)

    var diff = verify_states_equal(
        state_fused, state_baseline, 1e-10, "grid_fused", "scalar"
    )
    print("Cross-row HP fusion diff: " + String(diff))
    if diff < 1e-10:
        print("✓ PASSED")
    else:
        print("✗ FAILED")
