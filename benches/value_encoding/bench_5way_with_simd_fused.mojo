"""
5-way comparison: simd_v2, v_grid, fused_v3, v_grid_fused, simd_fused (prep + iqft).

Tests performance of all fusion strategies including the new clean simd_fused.
"""

from butterfly.utils.benchmark_runner import create_runner, should_autosave
from butterfly.core.state import QuantumState
from butterfly.core.execute_as_grid import execute_as_grid
from butterfly.core.execute_fused_v3 import execute_fused_v3
from butterfly.core.execute_v_grid_fused import execute_v_grid_fused
from butterfly.core.execute_simd_fused import execute_simd_fused
from butterfly.algos.value_encoding_circuit import encode_value_circuits_runtime
from collections import Dict, List


fn execute_simd_v2_circuit(params: Tuple[Int, Int]) raises -> QuantumState:
    """Execute using simd_v2 (verified baseline)."""
    var n = params[0]
    var circuit_idx = params[1]
    var circuits = encode_value_circuits_runtime(n, 4.7, swap=False)
    var state = QuantumState(n)

    circuits[circuit_idx].execute_simd_v2_dynamic(state)

    return state^


fn execute_v_grid_circuit(params: Tuple[Int, Int]) raises -> QuantumState:
    """Execute using Virtual Grid V5."""
    var n = params[0]
    var circuit_idx = params[1]
    var circuits = encode_value_circuits_runtime(n, 4.7, swap=False)
    var state = QuantumState(n)
    var col_bits = n - 3

    execute_as_grid(state, circuits[circuit_idx], col_bits)

    return state^


fn execute_fused_v3_circuit(params: Tuple[Int, Int]) raises -> QuantumState:
    """Execute using fused_v3."""
    var n = params[0]
    var circuit_idx = params[1]
    var circuits = encode_value_circuits_runtime(n, 4.7, swap=False)
    var state = QuantumState(n)

    if n == 15:
        execute_fused_v3[1 << 15](state, circuits[circuit_idx])
    elif n == 16:
        execute_fused_v3[1 << 16](state, circuits[circuit_idx])
    elif n == 17:
        execute_fused_v3[1 << 17](state, circuits[circuit_idx])
    elif n == 18:
        execute_fused_v3[1 << 18](state, circuits[circuit_idx])
    elif n == 19:
        execute_fused_v3[1 << 19](state, circuits[circuit_idx])
    elif n == 20:
        execute_fused_v3[1 << 20](state, circuits[circuit_idx])
    elif n == 21:
        execute_fused_v3[1 << 21](state, circuits[circuit_idx])
    elif n == 22:
        execute_fused_v3[1 << 22](state, circuits[circuit_idx])
    elif n == 23:
        execute_fused_v3[1 << 23](state, circuits[circuit_idx])
    elif n == 24:
        execute_fused_v3[1 << 24](state, circuits[circuit_idx])
    elif n == 25:
        execute_fused_v3[1 << 25](state, circuits[circuit_idx])
    elif n == 26:
        execute_fused_v3[1 << 26](state, circuits[circuit_idx])

    return state^


fn execute_v_grid_fused_circuit(params: Tuple[Int, Int]) raises -> QuantumState:
    """Execute using v_grid_fused (grid + fusion)."""
    var n = params[0]
    var circuit_idx = params[1]
    var circuits = encode_value_circuits_runtime(n, 4.7, swap=False)
    var state = QuantumState(n)
    var col_bits = n - 3

    execute_v_grid_fused(state, circuits[circuit_idx], col_bits)

    return state^


fn execute_simd_fused_circuit(params: Tuple[Int, Int]) raises -> QuantumState:
    """Execute using simd_fused (SIMD + fusion, clean implementation)."""
    var n = params[0]
    var circuit_idx = params[1]
    var circuits = encode_value_circuits_runtime(n, 4.7, swap=False)
    var state = QuantumState(n)

    execute_simd_fused(state, circuits[circuit_idx])

    return state^


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
    alias NAME = "bench_5way_with_simd_fused"
    alias DESCRIPTION = "5-Way: simd_v2, v_grid, fused_v3, v_grid_fused, simd_fused"

    var param_cols = List[String]("n", "circuit")
    var bench_cols = List[String](
        "simd_v2_ms",
        "v_grid_ms",
        "fused_v3_ms",
        "v_grid_fused_ms",
        "simd_fused_ms",
    )

    var runner = create_runner(NAME, DESCRIPTION, param_cols, bench_cols)

    print("5-Way Strategy Comparison (with simd_fused)")
    print("=" * 80)
    print("NEW: simd_fused - clean fusion implementation")
    print("GOAL: Compare all fusion strategies")
    print()

    var n_values = List[Int](15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26)
    var circuit_names = List[String]("prep", "iqft")

    # Verification phase
    print("Verification Phase")
    print("-" * 80)
    for i in range(len(n_values)):
        var n = n_values[i]

        for circuit_idx in range(2):
            var input = (n, circuit_idx)
            var circuit_name = circuit_names[circuit_idx]

            print(
                "Verifying n=" + String(n) + ", " + circuit_name + "...", end=""
            )

            # Always verify v_grid, v_grid_fused, and simd_fused vs simd_v2
            runner.verify(
                input,
                execute_simd_v2_circuit,
                execute_v_grid_circuit,
                compare_quantum_states,
                name1="simd_v2",
                name2="v_grid",
                tolerance=1e-10,
            )

            runner.verify(
                input,
                execute_simd_v2_circuit,
                execute_v_grid_fused_circuit,
                compare_quantum_states,
                name1="simd_v2",
                name2="v_grid_fused",
                tolerance=1e-10,
            )

            runner.verify(
                input,
                execute_simd_v2_circuit,
                execute_simd_fused_circuit,
                compare_quantum_states,
                name1="simd_v2",
                name2="simd_fused",
                tolerance=1e-10,
            )

            # Verify fused_v3 for all cases now that bug is fixed
            runner.verify(
                input,
                execute_simd_v2_circuit,
                execute_fused_v3_circuit,
                compare_quantum_states,
                name1="simd_v2",
                name2="fused_v3",
                tolerance=1e-10,
            )

            print(" (all verified)")

    print()
    print("✓ All verifications passed!")
    print()

    # Performance benchmarking phase
    print("Performance Benchmarking Phase")
    print("-" * 80)
    for i in range(len(n_values)):
        var n = n_values[i]

        for circuit_idx in range(2):
            var input = (n, circuit_idx)
            var params = Dict[String, String]()
            params["n"] = String(n)
            params["circuit"] = circuit_names[circuit_idx]

            runner.log_progress(
                "Benchmarking n="
                + String(n)
                + ", "
                + circuit_names[circuit_idx]
                + "..."
            )

            runner.add_perf_result(
                params, "simd_v2_ms", execute_simd_v2_circuit, input
            )
            runner.add_perf_result(
                params, "v_grid_ms", execute_v_grid_circuit, input
            )
            runner.add_perf_result(
                params, "fused_v3_ms", execute_fused_v3_circuit, input
            )
            runner.add_perf_result(
                params, "v_grid_fused_ms", execute_v_grid_fused_circuit, input
            )
            runner.add_perf_result(
                params, "simd_fused_ms", execute_simd_fused_circuit, input
            )

    print()
    runner.print_table(show_winner=True)
    runner.save_csv(NAME, autosave=should_autosave())

    print("\nStrategies:")
    print("  simd_v2_ms       = SIMD v2 (verified baseline)")
    print(
        "  v_grid_ms        = Virtual Grid V5 (grid parallelism + optimized"
        " controlled gates)"
    )
    print("  fused_v3_ms      = Fused V3 (gate fusion, has bugs)")
    print(
        "  v_grid_fused_ms  = v_grid + fusion (grid parallelism + fusion +"
        " optimized controlled gates)"
    )
    print("  simd_fused_ms    = SIMD + fusion (clean fusion, NEW!)")
    print("\nGoal: Compare all fusion strategies!")
