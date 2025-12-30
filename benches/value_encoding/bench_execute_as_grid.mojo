"""
Benchmark V5 strategies (v_grid and grid_fused_v4) vs SIMD v2 with verification.

Follows the prototypical pattern:
- Define execution functions with same signature (Tuple[Int, Int])
- Use runner.verify() for verification
- Use runner.add_perf_result() for automatic timing
- Parameters: n (qubit count), circuit_idx (0=Prep, 1=IQFT, 2=BitRev)
"""

from butterfly.utils.benchmark_runner import create_runner, should_autosave
from butterfly.core.state import QuantumState
from butterfly.core.execute_as_grid import execute_as_grid
from butterfly.core.execute_grid_fused_v4 import execute_grid_fused_v4
from butterfly.core.execute_fused_v3 import execute_fused_v3
from butterfly.algos.value_encoding_circuit import encode_value_circuits_runtime
from collections import Dict, List


# --- Execution Functions (all take Tuple[Int, Int]) ---


fn execute_simd_v2(params: Tuple[Int, Int]) raises -> QuantumState:
    """Execute using SIMD v2."""
    var n = params[0]
    var circuit_idx = params[1]
    var circuits = encode_value_circuits_runtime(n, 4.7, swap=True)
    var state = QuantumState(n)

    circuits[circuit_idx].execute_simd_v2_dynamic(state)

    return state^


fn execute_v_grid(params: Tuple[Int, Int]) raises -> QuantumState:
    """Execute using Virtual Grid V5."""
    var n = params[0]
    var circuit_idx = params[1]
    var circuits = encode_value_circuits_runtime(n, 4.7, swap=True)
    var state = QuantumState(n)
    var col_bits = n - 3

    execute_as_grid(state, circuits[circuit_idx], col_bits)

    return state^


fn execute_grid_fused_v4_wrapper(
    params: Tuple[Int, Int]
) raises -> QuantumState:
    """Execute using Grid Fused V4."""
    var n = params[0]
    var circuit_idx = params[1]
    var circuits = encode_value_circuits_runtime(n, 4.7, swap=True)
    var state = QuantumState(n)
    var col_bits = n - 3

    execute_grid_fused_v4(state, circuits[circuit_idx], col_bits)

    return state^


fn execute_fused_v3_wrapper(params: Tuple[Int, Int]) raises -> QuantumState:
    """Execute using Fused V3 (UNVERIFIED - has known bugs)."""
    var n = params[0]
    var circuit_idx = params[1]
    var circuits = encode_value_circuits_runtime(n, 4.7, swap=True)
    var state = QuantumState(n)

    # Compile-time dispatch
    if n == 15:
        execute_fused_v3[15](state, circuits[circuit_idx])
    elif n == 16:
        execute_fused_v3[16](state, circuits[circuit_idx])
    elif n == 17:
        execute_fused_v3[17](state, circuits[circuit_idx])
    elif n == 18:
        execute_fused_v3[18](state, circuits[circuit_idx])
    elif n == 19:
        execute_fused_v3[19](state, circuits[circuit_idx])
    elif n == 20:
        execute_fused_v3[20](state, circuits[circuit_idx])
    elif n == 21:
        execute_fused_v3[21](state, circuits[circuit_idx])
    elif n == 22:
        execute_fused_v3[22](state, circuits[circuit_idx])
    elif n == 23:
        execute_fused_v3[23](state, circuits[circuit_idx])
    elif n == 24:
        execute_fused_v3[24](state, circuits[circuit_idx])
    elif n == 25:
        execute_fused_v3[25](state, circuits[circuit_idx])
    elif n == 26:
        execute_fused_v3[26](state, circuits[circuit_idx])

    return state^


# --- Verification Comparator ---


fn compare_quantum_states(
    val1: QuantumState, val2: QuantumState, tolerance: Float64
) raises:
    """Specialized comparison for QuantumState."""
    if val1.size() != val2.size():
        raise Error("Verification failed: State sizes differ")

    var diff_sum = 0.0
    for i in range(val1.size()):
        var dr = val1.re[i] - val2.re[i]
        var di = val1.im[i] - val2.im[i]
        diff_sum += dr * dr + di * di

    if diff_sum > tolerance:
        raise Error(
            "Verification failed: QuantumStates differ by " + String(diff_sum)
        )


fn main() raises:
    alias NAME = "bench_execute_as_grid"
    alias DESCRIPTION = "V5 Strategies vs SIMD_V2 (Verified) + Fused V3 (Unverified)"

    # Define columns
    var param_cols = List[String]("n", "circuit")
    var bench_cols = List[String](
        "simd_v2_ms", "v_grid_ms", "grid_fused_v4_ms", "fused_v3_ms"
    )

    # Create runner
    var runner = create_runner(NAME, DESCRIPTION, param_cols, bench_cols)

    print("Benchmarking Virtual Grid V5 vs SIMD v2 (with verification)")
    print("=" * 80)
    print()

    var n_values = List[Int](15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26)
    var circuit_names = List[String]("prep", "iqft", "bit_rev")

    # Verification phase
    print("Verification Phase")
    print("-" * 80)
    for i in range(len(n_values)):
        var n = n_values[i]
        for circuit_idx in range(3):
            var input = (n, circuit_idx)

            runner.verify(
                input,
                execute_simd_v2,
                execute_v_grid,
                compare_quantum_states,
                name1="simd_v2",
                name2="v_grid",
                tolerance=1e-10,
            )

            runner.verify(
                input,
                execute_simd_v2,
                execute_grid_fused_v4_wrapper,
                compare_quantum_states,
                name1="simd_v2",
                name2="grid_fused_v4",
                tolerance=1e-10,
            )

    print("✓ All verifications passed!")
    print()

    # Performance benchmarking phase
    print("Performance Benchmarking Phase")
    print("-" * 80)
    for i in range(len(n_values)):
        var n = n_values[i]
        for circuit_idx in range(len(circuit_names)):
            var input = (n, circuit_idx)
            var params = Dict[String, String]()
            params["n"] = String(n)
            params["circuit"] = circuit_names[circuit_idx]

            runner.log_progress(
                "Benchmarking n="
                + String(n)
                + ", circuit="
                + circuit_names[circuit_idx]
                + "..."
            )

            runner.add_perf_result(params, "simd_v2_ms", execute_simd_v2, input)
            runner.add_perf_result(params, "v_grid_ms", execute_v_grid, input)
            runner.add_perf_result(
                params, "grid_fused_v4_ms", execute_grid_fused_v4_wrapper, input
            )
            runner.add_perf_result(
                params, "fused_v3_ms", execute_fused_v3_wrapper, input
            )

    print()
    runner.print_table(show_winner=True)
    runner.save_csv(NAME, autosave=should_autosave())

    print("\nStrategies:")
    print("  simd_v2_ms       = SIMD v2 dynamic (verified)")
    print("  v_grid_ms        = Virtual Grid V5 (verified)")
    print("  grid_fused_v4_ms = Grid Fused V4 (verified)")
    print("  fused_v3_ms      = Fused V3 (UNVERIFIED - has scaling bug)")
    print("\nCircuits:")
    print("  prep    = Preparation phase (Circuit 0)")
    print("  iqft    = IQFT phase (Circuit 1)")
    print("  bit_rev = Bit Reversal (Circuit 2)")
