"""
Comprehensive benchmark suite comparing all Mojo strategies and Qiskit.
Strategies: scalar, simd_v2, fused_v3, v_grid, v_grid_fused, qiskit.
Circuits: prep and prep+iqft (from value encoding).
"""

from butterfly.utils.benchmark_runner import create_runner, should_autosave
from butterfly.core.state import QuantumState
from butterfly.core.circuit import QuantumCircuit
from butterfly.core.execute_as_grid import execute_as_grid
from butterfly.core.execute_v_grid_fused import execute_v_grid_fused
from butterfly.algos.value_encoding_circuit import encode_value_circuits_runtime
from butterfly.utils.visualization import print_state
from collections import Dict, List
from math import pi


fn execute_v_grid_circuit_8rows(circuit: QuantumCircuit) raises -> QuantumState:
    """Execute using Virtual Grid V5 (row-local parallelism)."""
    var n = circuit.num_qubits
    var state = QuantumState(n)

    var col_bits = n - 3  # Typical heuristic
    execute_as_grid(state, circuit, col_bits)

    return state^


fn execute_v_grid_circuit_rows(
    circuit: QuantumCircuit,
) raises -> QuantumState:
    """Execute using Virtual Grid V5 (row-local parallelism)."""
    var n = circuit.num_qubits
    var state = QuantumState(n)

    var col_bits = n - 3  # Typical heuristic
    execute_as_grid(state, circuit, col_bits)

    return state^


fn execute_v_grid_fused_circuit_8rows(
    circuit: QuantumCircuit,
) raises -> QuantumState:
    """Execute using Virtual Grid + Fusion."""
    var n = circuit.num_qubits
    var state = QuantumState(n)

    var col_bits = n - 3
    execute_v_grid_fused(state, circuit, col_bits)

    return state^


# --- Common Utilities ---


fn compare_quantum_states(
    val1: QuantumState, val2: QuantumState, tolerance: Float64
) raises:
    """Comparison hook for verification."""
    if val1.size() != val2.size():
        raise Error("Verification failed: State sizes differ")

    var diff_sum = 0.0
    for i in range(val1.size()):
        var dr = val1.re[i] - val2.re[i]
        var di = val1.im[i] - val2.im[i]
        diff_sum += dr * dr + di * di

    if diff_sum > tolerance:
        print_state(val1)
        print_state(val2)
        raise Error("Verification failed: State diff " + String(diff_sum))


fn main() raises:
    alias NAME = "grid_row_comparison"
    alias DESCRIPTION = "Value Encoding: Virtual Grid vs Virtual Grid + Fusion"

    var p_cols = List[String]("n", "circuit")
    var b_cols = List[String](
        "v_grid_8rows",
        "v_grid_fused_8rows",
    )

    var runner = create_runner(NAME, DESCRIPTION, p_cols, b_cols, 6)

    var n_values = List[Int]()
    for n in range(10, 25):
        n_values.append(n)
    var circuits = List[String]("prep", "prep+iqft")

    for i in range(len(n_values)):
        var n = n_values[i]

        # Pre-generate circuits for this n to avoid overhead in executors
        var full_circuits = encode_value_circuits_runtime(n, 4.7, swap=False)

        for i in [0, 1]:
            var input = full_circuits[0].copy()
            for j in range(1, i + 1):
                input.append_circuit(full_circuits[j])
            var c_name = circuits[i]

            print("\nCase: n=" + String(n) + ", circuit=" + c_name)

            var params = Dict[String, String]()
            params["n"] = String(n)
            params["circuit"] = c_name

            if n <= 22:
                # Mojo-to-Mojo parity

                runner.verify(
                    input,
                    execute_v_grid_circuit_8rows,
                    execute_v_grid_circuit_rows,
                    compare_quantum_states,
                    name1="v_grid_8rows",
                    name2="v_grid_rows",
                )

            # 2. Performance Phase
            runner.add_perf_result(
                params, "v_grid_8rows", execute_v_grid_circuit_8rows, input
            )
            runner.add_perf_result(
                params, "v_grid_rows", execute_v_grid_circuit_rows, input
            )
            runner.add_perf_result(
                params,
                "v_grid_fused_8rows",
                execute_v_grid_fused_circuit_8rows,
                input,
            )

    runner.print_table()
    runner.save_csv(NAME, autosave=should_autosave())
