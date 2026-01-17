"""Test cross-row HP, HH, and PP fusion correctness."""
from butterfly.core.state import QuantumState
from butterfly.core.quantum_circuit import QuantumCircuit
from butterfly.core.executors import execute_scalar, execute_grid_fused
from butterfly.utils.benchmark_verify import verify_states_equal
from butterfly.utils.context import ExecContext
from math import pi


fn test_cross_row_hp() raises -> FloatType:
    """Test cross-row H+P fusion."""
    var n = 12
    var col_bits = 6

    var qc = QuantumCircuit(n)
    for i in range(3):
        qc.h(i)
    qc.h(col_bits)  # Cross-row H
    qc.p(col_bits + 1, pi / 4)  # Cross-row P

    var state_baseline = QuantumState(n)
    execute_scalar(state_baseline, qc)

    var state_fused = QuantumState(n)
    var ctx = ExecContext()
    execute_grid_fused(state_fused, qc, ctx)

    return verify_states_equal(
        state_fused, state_baseline, 1e-10, "grid_fused", "scalar"
    )


fn test_cross_row_hh() raises -> FloatType:
    """Test cross-row H+H fusion."""
    var n = 12
    var col_bits = 6

    var qc = QuantumCircuit(n)
    for i in range(3):
        qc.h(i)
    qc.h(col_bits)  # Cross-row H
    qc.h(col_bits + 1)  # Cross-row H

    var state_baseline = QuantumState(n)
    execute_scalar(state_baseline, qc)

    var state_fused = QuantumState(n)
    var ctx = ExecContext()
    execute_grid_fused(state_fused, qc, ctx)

    return verify_states_equal(
        state_fused, state_baseline, 1e-10, "grid_fused", "scalar"
    )


fn test_cross_row_pp() raises -> FloatType:
    """Test cross-row P+P fusion."""
    var n = 12
    var col_bits = 6

    var qc = QuantumCircuit(n)
    for i in range(3):
        qc.h(i)
    qc.p(col_bits, pi / 4)  # Cross-row P
    qc.p(col_bits + 1, pi / 3)  # Cross-row P

    var state_baseline = QuantumState(n)
    execute_scalar(state_baseline, qc)

    var state_fused = QuantumState(n)
    var ctx = ExecContext()
    execute_grid_fused(state_fused, qc, ctx)

    return verify_states_equal(
        state_fused, state_baseline, 1e-10, "grid_fused", "scalar"
    )


fn main() raises:
    print("Testing cross-row fusion...")

    var hp_diff = test_cross_row_hp()
    print("Cross-row HP fusion diff: " + String(hp_diff))
    if hp_diff < 1e-10:
        print("✓ HP PASSED")
    else:
        print("✗ HP FAILED")

    var hh_diff = test_cross_row_hh()
    print("Cross-row HH fusion diff: " + String(hh_diff))
    if hh_diff < 1e-10:
        print("✓ HH PASSED")
    else:
        print("✗ HH FAILED")

    var pp_diff = test_cross_row_pp()
    print("Cross-row PP fusion diff: " + String(pp_diff))
    if pp_diff < 1e-10:
        print("✓ PP PASSED")
    else:
        print("✗ PP FAILED")
