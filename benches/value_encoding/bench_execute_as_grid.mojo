"""
Benchmark execute_as_grid (Virtual Grid) vs other strategies.
"""
from butterfly.core.state import QuantumState
from butterfly.core.circuit import QuantumCircuit, execute_simd_v2
from butterfly.core.execute_fused_v3 import execute_fused_v3
from butterfly.core.execute_as_grid import execute_as_grid
from butterfly.core.execute_grid_fused_v4 import execute_grid_fused_v4
from butterfly.algos.value_encoding_circuit import encode_value_circuits_runtime
from butterfly.utils.benchmark_runner import create_runner, should_autosave
from collections import Dict, List
from math import pi

# --- Benchmark Functions ---


fn run_generic(n: Int) raises -> QuantumState:
    var circuits = encode_value_circuits_runtime(n, 4.7)
    var state = QuantumState(n)
    circuits[0].execute(state)
    return state^


fn run_simd_v2(n: Int) raises -> QuantumState:
    var circuits = encode_value_circuits_runtime(n, 4.7)
    var state = QuantumState(n)
    circuits[0].execute_simd_v2_dynamic(state)
    return state^


fn run_fused_v3(n: Int) raises -> QuantumState:
    var circuits = encode_value_circuits_runtime(n, 4.7)
    var state = QuantumState(n)
    circuits[0].execute_fused_v3_dynamic(state)
    return state^


fn run_virtual_grid_8row(n: Int) raises -> QuantumState:
    var circuits = encode_value_circuits_runtime(n, 4.7)
    var state = QuantumState(n)
    # Virtual Grid with 8 rows means col_bits = n - 3
    execute_as_grid(state, circuits[0], n - 3)
    return state^


fn run_grid_fused_v4_8row(n: Int) raises -> QuantumState:
    var circuits = encode_value_circuits_runtime(n, 4.7)
    var state = QuantumState(n)
    # Grid Fused with 8 rows
    execute_grid_fused_v4(state, circuits[0], n - 3)
    return state^


# --- Verifier ---


fn compare_quantum_states(
    val1: QuantumState, val2: QuantumState, tolerance: Float64
) raises:
    """Specialized comparison for QuantumState."""
    if val1.size() != val2.size():
        raise Error("Verification failed: State sizes differ")

    for i in range(val1.size()):
        var dr = val1.re[i] - val2.re[i]
        var di = val1.im[i] - val2.im[i]
        if (dr * dr + di * di) > tolerance:
            raise Error(
                "Verification failed: QuantumStates differ at index "
                + String(i)
            )


fn main() raises:
    alias NAME = "bench_execute_as_grid"
    alias DESCRIPTION = "Generic vs SIMD_V2 vs Fused_V3 vs Virtual Grid"

    var param_cols = List[String]("n")
    var bench_cols = List[String](
        "generic", "simd_v2", "fused_v3", "v_grid_8row", "v_gfused_v4"
    )

    var runner = create_runner(NAME, DESCRIPTION, param_cols, bench_cols)

    # Verification phase
    for n in range(4, 11):
        runner.verify(
            n,
            run_simd_v2,
            run_virtual_grid_8row,
            compare=compare_quantum_states,
            name1="simd_v2",
            name2="v_grid_8row",
            tolerance=1e-10,
        )

    # Performance phase
    for n in range(15, 27):
        var params = Dict[String, String]()
        params["n"] = String(n)

        runner.log_progress("Gathering n=" + String(n))
        runner.add_perf_result(params, "generic", run_generic, n)
        runner.add_perf_result(params, "simd_v2", run_simd_v2, n)
        runner.add_perf_result(params, "fused_v3", run_fused_v3, n)
        runner.add_perf_result(params, "v_grid_8row", run_virtual_grid_8row, n)
        runner.add_perf_result(params, "v_gfused_v4", run_grid_fused_v4_8row, n)

    runner.print_table(show_winner=True)
    runner.save_csv(NAME, autosave=should_autosave())
