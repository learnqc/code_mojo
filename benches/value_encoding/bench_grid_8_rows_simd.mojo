"""
Focused benchmark: Generic vs SIMD v2 vs Grid 8-row with SIMD.

Tests the three key strategies across n=3-27 for value encoding circuits.
"""
# pixi run mojo run -I . benches/value_encoding/bench_grid_8_rows_simd.mojo

from butterfly.utils.benchmark_runner import create_runner
from butterfly.core.grid_state import GridQuantumState
from butterfly.core.state import QuantumState
from butterfly.core.types import FloatType
from butterfly.algos.value_encoding_circuit import encode_value_circuits_runtime
from collections import Dict, List


fn execute_generic(params: Tuple[Int, FloatType]) raises -> QuantumState:
    """Execute using standard execute()."""
    var n = params[0]
    var value = params[1]
    var circuits = encode_value_circuits_runtime(n, value, True)[0:1]
    var state = QuantumState(n)
    for i in range(len(circuits)):
        circuits[i].execute(state)
    return state^


fn execute_simd_v2(params: Tuple[Int, FloatType]) raises -> QuantumState:
    """Execute using SIMD v2."""
    var n = params[0]
    var value = params[1]
    var circuits = encode_value_circuits_runtime(n, value, True)[0:1]
    var state = QuantumState(n)
    for i in range(len(circuits)):
        circuits[i].execute_simd_v2_dynamic(state)
    return state^


fn execute_grid_8row_simd(params: Tuple[Int, FloatType]) raises -> QuantumState:
    """Execute using GridQuantumState with 8 rows and SIMD."""
    var n = params[0]
    var value = params[1]
    var circuits = encode_value_circuits_runtime(n, value, True)[0:1]
    var grid = GridQuantumState(n, 3)
    for i in range(len(circuits)):
        grid.execute[True](circuits[i])

    # Convert to QuantumState
    var state = QuantumState(n)
    for i in range(grid.size()):
        state.re[i] = grid.re[i]
        state.im[i] = grid.im[i]
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
    alias NAME = "grid_8_rows_simd"
    alias DESCRIPTION = "Generic vs SIMD v2 vs Grid 8-row SIMD (n=3-27)"

    var param_cols = List[String]("n", "value")
    var bench_cols = List[String](
        "generic_ms", "simd_v2_ms", "grid_8row_simd_ms"
    )

    var runner = create_runner(NAME, DESCRIPTION, param_cols, bench_cols)

    print("Benchmarking: Generic vs SIMD v2 vs Grid 8-row SIMD")
    print("=" * 80)
    print()

    alias value = 4.7
    var n_values = List[Int](3)

    # Verification phase
    print("Verification Phase")
    print("-" * 80)
    for i in range(len(n_values)):
        var n = n_values[i]
        var input = (n, value)

        print("Verifying n=" + String(n) + "...")

        runner.verify(
            input,
            execute_generic,
            execute_simd_v2,
            compare_quantum_states,
            name1="generic",
            name2="simd_v2",
            tolerance=1e-10,
        )

        runner.verify(
            input,
            execute_generic,
            execute_grid_8row_simd,
            compare_quantum_states,
            name1="generic",
            name2="grid_8row_simd",
            tolerance=1e-10,
        )

    print("✓ All verifications passed!")
    print()

    # Performance benchmarking phase
    print("Performance Benchmarking Phase")
    print("-" * 80)
    for i in range(len(n_values)):
        var n = n_values[i]
        var input = (n, value)
        var params = Dict[String, String]()
        params["n"] = String(n)
        params["value"] = String(value)

        runner.log_progress("Benchmarking n=" + String(n) + "...")

        runner.add_perf_result(params, "generic_ms", execute_generic, input)
        runner.add_perf_result(params, "simd_v2_ms", execute_simd_v2, input)
        runner.add_perf_result(
            params, "grid_8row_simd_ms", execute_grid_8row_simd, input
        )

    print()
    runner.print_table(show_winner=True)
    runner.save_csv(NAME)

    print("\nStrategies:")
    print("  generic_ms       = Standard execute()")
    print("  simd_v2_ms       = SIMD v2 dynamic")
    print("  grid_8row_simd_ms = GridQuantumState 8 rows with SIMD")
