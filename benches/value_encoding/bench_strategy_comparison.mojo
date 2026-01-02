"""
Comprehensive benchmark suite comparing all Mojo strategies and Qiskit.
Strategies: scalar, simd_v2, fused_v3, v_grid, v_grid_fused, qiskit.
Circuits: prep and prep+iqft (from value encoding).
"""

from butterfly.utils.benchmark_runner import (
    create_runner,
    should_autosave,
    LabeledFunction,
)
from butterfly.core.state import QuantumState
from butterfly.core.circuit import QuantumCircuit
from butterfly.core.execute_as_grid import execute_as_grid
from butterfly.core.execute_fused_v3 import execute_fused_v3
from butterfly.core.execute_v_grid_fused import execute_v_grid_fused
from butterfly.core.execute_grid_fused import execute_grid_fused
from butterfly.algos.value_encoding_circuit import encode_value_circuits_runtime
from butterfly.utils.quantum_interop import (
    get_qiskit_prep_state,
    get_qiskit_full_state,
)
from butterfly.utils.visualization import print_state
from collections import Dict, List
from math import pi

alias Executor = LabeledFunction[QuantumCircuit, QuantumState]
alias encoding_value = 4.7

# --- Baseline Executors ---


fn execute_scalar_circuit(circuit: QuantumCircuit) raises -> QuantumState:
    """Execute using generic (scalar) strategy."""
    var n = circuit.num_qubits
    var state = QuantumState(n)

    from butterfly.core.state import (
        transform,
        c_transform,
        mc_transform,
        bit_reverse_state,
    )
    from butterfly.core.circuit import (
        GateTransformation,
        SingleControlGateTransformation,
        MultiControlGateTransformation,
        BitReversalTransformation,
    )

    for i in range(len(circuit.transformations)):
        var t = circuit.transformations[i]
        if t.isa[GateTransformation]():
            var g = t[GateTransformation].copy()
            transform(state, g.target, g.gate)
        elif t.isa[SingleControlGateTransformation]():
            var g = t[SingleControlGateTransformation].copy()
            c_transform(state, g.control, g.target, g.gate)
        elif t.isa[MultiControlGateTransformation]():
            var g = t[MultiControlGateTransformation].copy()
            mc_transform(state, g.controls, g.target, g.gate)
        elif t.isa[BitReversalTransformation]():
            bit_reverse_state(state)

    return state^


fn execute_simd_v2_circuit(circuit: QuantumCircuit) raises -> QuantumState:
    """Execute using simd_v2 (naive parallel SIMD)."""
    var n = circuit.num_qubits
    var state = QuantumState(n)

    var circ = circuit.copy()
    circ.execute_simd_v2_dynamic(state)

    return state^


fn execute_fused_v3_circuit(circuit: QuantumCircuit) raises -> QuantumState:
    """Execute using fused_v3 (gate fusion + cache blocking)."""
    var n = circuit.num_qubits
    var state = QuantumState(n)

    if n == 3:
        execute_fused_v3[1 << 3](state, circuit)
    elif n == 4:
        execute_fused_v3[1 << 4](state, circuit)
    elif n == 5:
        execute_fused_v3[1 << 5](state, circuit)
    elif n == 6:
        execute_fused_v3[1 << 6](state, circuit)
    elif n == 7:
        execute_fused_v3[1 << 7](state, circuit)
    elif n == 8:
        execute_fused_v3[1 << 8](state, circuit)
    elif n == 9:
        execute_fused_v3[1 << 9](state, circuit)
    elif n == 10:
        execute_fused_v3[1 << 10](state, circuit)
    elif n == 11:
        execute_fused_v3[1 << 11](state, circuit)
    elif n == 12:
        execute_fused_v3[1 << 12](state, circuit)
    elif n == 13:
        execute_fused_v3[1 << 13](state, circuit)
    elif n == 14:
        execute_fused_v3[1 << 14](state, circuit)
    elif n == 15:
        execute_fused_v3[1 << 15](state, circuit)
    elif n == 16:
        execute_fused_v3[1 << 16](state, circuit)
    elif n == 17:
        execute_fused_v3[1 << 17](state, circuit)
    elif n == 18:
        execute_fused_v3[1 << 18](state, circuit)
    elif n == 19:
        execute_fused_v3[1 << 19](state, circuit)
    elif n == 20:
        execute_fused_v3[1 << 20](state, circuit)
    elif n == 21:
        execute_fused_v3[1 << 21](state, circuit)
    elif n == 22:
        execute_fused_v3[1 << 22](state, circuit)
    elif n == 23:
        execute_fused_v3[1 << 23](state, circuit)
    elif n == 24:
        execute_fused_v3[1 << 24](state, circuit)
    elif n == 25:
        execute_fused_v3[1 << 25](state, circuit)
    elif n == 26:
        execute_fused_v3[1 << 26](state, circuit)
    elif n == 27:
        execute_fused_v3[1 << 27](state, circuit)

    return state^


fn execute_v_grid_circuit(circuit: QuantumCircuit) raises -> QuantumState:
    """Execute using Virtual Grid V5 (row-local parallelism)."""
    var n = circuit.num_qubits
    var state = QuantumState(n)

    var col_bits = n - 3  # Typical heuristic
    execute_as_grid(state, circuit, col_bits)

    return state^


fn execute_v_grid_fused_circuit(circuit: QuantumCircuit) raises -> QuantumState:
    """Execute using Virtual Grid + Fusion."""
    var n = circuit.num_qubits
    var state = QuantumState(n)

    var col_bits = n - 3
    execute_v_grid_fused(state, circuit, col_bits)

    return state^


fn execute_grid_fusion_circuit(circuit: QuantumCircuit) raises -> QuantumState:
    """Execute using Grid-Fusion (new implementation)."""
    var n = circuit.num_qubits
    var state = QuantumState(n)
    var col_bits = n - 3
    var mut_circ = circuit.copy()

    if n == 3:
        execute_grid_fused[1 << 3, 8](state, mut_circ, col_bits)
    elif n == 4:
        execute_grid_fused[1 << 4, 8](state, mut_circ, col_bits)
    elif n == 5:
        execute_grid_fused[1 << 5, 8](state, mut_circ, col_bits)
    elif n == 6:
        execute_grid_fused[1 << 6, 8](state, mut_circ, col_bits)
    elif n == 7:
        execute_grid_fused[1 << 7, 8](state, mut_circ, col_bits)
    elif n == 8:
        execute_grid_fused[1 << 8, 8](state, mut_circ, col_bits)
    elif n == 9:
        execute_grid_fused[1 << 9, 8](state, mut_circ, col_bits)
    elif n == 10:
        execute_grid_fused[1 << 10, 8](state, mut_circ, col_bits)
    elif n == 11:
        execute_grid_fused[1 << 11, 8](state, mut_circ, col_bits)
    elif n == 12:
        execute_grid_fused[1 << 12, 8](state, mut_circ, col_bits)
    elif n == 13:
        execute_grid_fused[1 << 13, 8](state, mut_circ, col_bits)
    elif n == 14:
        execute_grid_fused[1 << 14, 8](state, mut_circ, col_bits)
    elif n == 15:
        execute_grid_fused[1 << 15, 8](state, mut_circ, col_bits)
    elif n == 16:
        execute_grid_fused[1 << 16, 8](state, mut_circ, col_bits)
    elif n == 17:
        execute_grid_fused[1 << 17, 8](state, mut_circ, col_bits)
    elif n == 18:
        execute_grid_fused[1 << 18, 8](state, mut_circ, col_bits)
    elif n == 19:
        execute_grid_fused[1 << 19, 8](state, mut_circ, col_bits)
    elif n == 20:
        execute_grid_fused[1 << 20, 8](state, mut_circ, col_bits)
    elif n == 21:
        execute_grid_fused[1 << 21, 8](state, mut_circ, col_bits)
    elif n == 22:
        execute_grid_fused[1 << 22, 8](state, mut_circ, col_bits)
    elif n == 23:
        execute_grid_fused[1 << 23, 8](state, mut_circ, col_bits)
    elif n == 24:
        execute_grid_fused[1 << 24, 8](state, mut_circ, col_bits)
    elif n == 25:
        execute_grid_fused[1 << 25, 8](state, mut_circ, col_bits)
    elif n == 26:
        execute_grid_fused[1 << 26, 8](state, mut_circ, col_bits)
    elif n == 27:
        execute_grid_fused[1 << 27, 8](state, mut_circ, col_bits)

    return state^


# --- Qiskit Interop Executors ---


fn execute_qiskit_circuit(circuit: QuantumCircuit) raises -> QuantumState:
    """Fetch current state from Qiskit (baseline)."""
    var n = circuit.num_qubits
    var value = encoding_value

    # Heuristic: prep circuit has exactly 2*n gates (n H + n P)
    if len(circuit.transformations) <= 2 * n:
        return get_qiskit_prep_state(n, value)
    else:
        return get_qiskit_full_state(n, value)


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
    from python import Python

    try:
        var sys = Python.import_module("sys")
        _ = sys.path.append(".")
    except:
        pass

    alias NAME = "value_encoding_strategies"
    alias DESCRIPTION = "Value Encoding: Mojo vs Qiskit "

    var p_cols = List[String]("n", "circuit")
    var executors = List[Executor]()
    executors.append(Executor("scalar", execute_scalar_circuit))
    executors.append(Executor("simd_v2", execute_simd_v2_circuit))
    executors.append(Executor("fused_v3", execute_fused_v3_circuit))
    executors.append(Executor("v_grid", execute_v_grid_circuit))
    executors.append(Executor("v_grid_fused", execute_v_grid_fused_circuit))
    executors.append(Executor("grid_fusion", execute_grid_fusion_circuit))
    executors.append(Executor("qiskit", execute_qiskit_circuit))

    var qiskit_executors = List[String]("q_aer_raw", "q_latency", "q_aer_opt")
    var qiskit_executor_modes = List[String]("none", "in_loop", "cached")

    var excluded_executors = List[String]("scalar", "qiskit")
    var excluded_qiskit_executors = List[String]("q_aer_raw", "q_latency")

    var b_cols = List[String]()
    for i in range(len(executors)):
        b_cols.append(executors[i].name)

    for i in range(len(qiskit_executors)):
        b_cols.append(qiskit_executors[i])

    var runner = create_runner(NAME, DESCRIPTION, p_cols, b_cols, 9)

    var n_range = range(12, 27)
    var n_verification_range = range(22)
    var n_verification_list = List[Int]()
    for n in n_verification_range:
        n_verification_list.append(n)
    var n_exclusion_range = range(20, 27)
    var n_exclusion_list = List[Int]()
    for n in n_exclusion_range:
        n_exclusion_list.append(n)

    var n_values = List[Int]()
    for n in n_range:
        n_values.append(n)
    var circuits = List[String]("prep", "prep+iqft")

    for i in range(len(n_values)):
        var n = n_values[i]

        # Pre-generate circuits for this n to avoid overhead in executors
        var full_circuits = encode_value_circuits_runtime(
            n, encoding_value, swap=False
        )

        for i in [0, 1]:
            var input = full_circuits[0].copy()
            for j in range(1, i + 1):
                input.append_circuit(full_circuits[j])
            var c_name = circuits[i]

            print("\nCase: n=" + String(n) + ", circuit=" + c_name)

            var params = Dict[String, String]()
            params["n"] = String(n)
            params["circuit"] = c_name

            # 1. Verification Phase
            if n in n_verification_list:
                # Comprehensive Parity Check
                runner.log_progress(
                    "Checking parity (Mojo vs Qiskit reference)..."
                )
                runner.verify(
                    input,
                    executors,
                    compare_quantum_states,
                    stop_on_failure=False,
                )

            # 2. Performance Measurement
            if len(params) > 0:
                if n in n_exclusion_list:
                    var filtered_executors = List[Executor]()
                    for idx in range(len(executors)):
                        var name = executors[idx].name
                        if name not in excluded_executors:
                            filtered_executors.append(executors[idx])
                    runner.add_perf_results(params, filtered_executors, input)
                else:
                    runner.add_perf_results(params, executors, input)

            # Qiskit time (via interop wrapper)
            from python import Python

            var py_bench = Python.import_module(
                "butterfly.utils.external_benchmarks"
            )

            if i == 0:
                for q_executor, q_mode in zip(
                    qiskit_executors, qiskit_executor_modes
                ):
                    if q_executor not in excluded_qiskit_executors:
                        var result = atof(
                            String(
                                py_bench.benchmark_qiskit_prep(
                                    n, encoding_value, iters=3, mode=q_mode
                                )
                            )
                        )
                        runner.add_result(params, q_executor, result)

            else:
                for q_executor, q_mode in zip(
                    qiskit_executors, qiskit_executor_modes
                ):
                    if q_executor not in excluded_qiskit_executors:
                        var result = atof(
                            String(
                                py_bench.benchmark_qiskit_value_encoding(
                                    n, encoding_value, iters=3, mode=q_mode
                                )
                            )
                        )
                        runner.add_result(params, q_executor, result)

    runner.print_table()
    runner.save_csv(NAME, autosave=should_autosave())
