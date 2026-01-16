"""
Test L2 cache tiling correctness for grid column operations.

Verifies that the new tiled column kernels produce identical results
to the original scalar implementation (which is known correct).
"""
from butterfly.core.state import QuantumState
from butterfly.core.quantum_circuit import QuantumCircuit
from butterfly.core.gates import H_Gate, P_Gate
from butterfly.core.executors import execute_scalar
from butterfly.core.transformations_grid import transform_grid, L2_TILE_COLS
from butterfly.utils.context import ExecContext
from butterfly.utils.benchmark_verify import verify_states_equal
from math import pi, log2


fn test_grid_h_cross_row(n: Int, col_bits: Int) raises:
    """Test H gate applied to a qubit in the 'row' region (cross-row operation).
    """
    # Target a qubit beyond col_bits (forces column operation with tiling)
    var target = col_bits  # First row qubit

    # Create test circuit with H gates
    var qc = QuantumCircuit(n)
    # Initialize with superposition on first few qubits
    for i in range(min(3, n)):
        qc.h(i)
    # Apply H to cross-row target
    qc.h(target)

    # Execute with scalar as baseline (known correct)
    var state_baseline = QuantumState(n)
    execute_scalar(state_baseline, qc)

    # Manually apply transform_grid for the cross-row H
    var state_tiled = QuantumState(n)
    # First apply the initial H gates (within row)
    for i in range(min(3, n)):
        var ctx = ExecContext()
        ctx.grid_use_parallel = False
        transform_grid(state_tiled, col_bits, i, H_Gate, ctx)
    # Then apply the cross-row H (this uses tiling)
    var ctx2 = ExecContext()
    ctx2.grid_use_parallel = False
    transform_grid(state_tiled, col_bits, target, H_Gate, ctx2)

    # Compare
    var diff = verify_states_equal(
        state_tiled, state_baseline, 1e-10, "grid_tiled", "scalar_baseline"
    )
    print(
        "  n="
        + String(n)
        + ", col_bits="
        + String(col_bits)
        + ", target="
        + String(target)
        + " (H) -> diff="
        + String(diff)
        + " ✓"
    )


fn test_grid_p_cross_row(n: Int, col_bits: Int) raises:
    """Test P gate applied to a qubit in the 'row' region."""
    var target = col_bits
    var theta = pi / 4.0

    # Create test circuit
    var qc = QuantumCircuit(n)
    for i in range(min(3, n)):
        qc.h(i)
    qc.p(target, theta)

    # Baseline with scalar
    var state_baseline = QuantumState(n)
    execute_scalar(state_baseline, qc)

    # Manual grid execution
    var state_tiled = QuantumState(n)
    for i in range(min(3, n)):
        var ctx = ExecContext()
        ctx.grid_use_parallel = False
        transform_grid(state_tiled, col_bits, i, H_Gate, ctx)
    var ctx2 = ExecContext()
    ctx2.grid_use_parallel = False
    transform_grid(state_tiled, col_bits, target, P_Gate(theta), ctx2)

    var diff = verify_states_equal(
        state_tiled, state_baseline, 1e-10, "grid_tiled_P", "scalar_baseline"
    )
    print(
        "  n="
        + String(n)
        + ", col_bits="
        + String(col_bits)
        + ", target="
        + String(target)
        + " (P) -> diff="
        + String(diff)
        + " ✓"
    )


fn test_multiple_cross_row(n: Int, col_bits: Int) raises:
    """Test multiple cross-row gates (IQFT-like pattern)."""
    var qc = QuantumCircuit(n)

    # Initialize with Hadamards
    for i in range(n):
        qc.h(i)

    # Apply cross-row operations
    for target in range(col_bits, min(col_bits + 2, n)):
        qc.h(target)
        qc.p(target, pi / 4)

    # Baseline
    var state_baseline = QuantumState(n)
    execute_scalar(state_baseline, qc)

    # Manual grid
    var state_tiled = QuantumState(n)
    var ctx = ExecContext()
    ctx.grid_use_parallel = False

    # Apply all H gates
    for i in range(n):
        transform_grid(state_tiled, col_bits, i, H_Gate, ctx)

    # Apply cross-row H and P
    for target in range(col_bits, min(col_bits + 2, n)):
        transform_grid(state_tiled, col_bits, target, H_Gate, ctx)
        transform_grid(state_tiled, col_bits, target, P_Gate(pi / 4), ctx)

    var diff = verify_states_equal(
        state_tiled, state_baseline, 1e-10, "grid_multi", "scalar_baseline"
    )
    print(
        "  n="
        + String(n)
        + ", col_bits="
        + String(col_bits)
        + ", multi-gate -> diff="
        + String(diff)
        + " ✓"
    )


fn main() raises:
    print("=" * 60)
    print("L2 Cache Tiling Verification Test")
    print("L2_TILE_COLS = " + String(L2_TILE_COLS))
    print("=" * 60)

    print("\n[1] H gate cross-row (tiled H kernel):")
    test_grid_h_cross_row(n=10, col_bits=5)
    test_grid_h_cross_row(n=12, col_bits=6)
    test_grid_h_cross_row(n=14, col_bits=7)

    print("\n[2] P gate cross-row (tiled generic kernel):")
    test_grid_p_cross_row(n=10, col_bits=5)
    test_grid_p_cross_row(n=12, col_bits=6)
    test_grid_p_cross_row(n=14, col_bits=7)

    print("\n[3] Multiple cross-row gates:")
    test_multiple_cross_row(n=10, col_bits=5)
    test_multiple_cross_row(n=12, col_bits=6)
    test_multiple_cross_row(n=14, col_bits=7)

    print("\n" + "=" * 60)
    print("All L2 tiling tests PASSED!")
    print("=" * 60)
