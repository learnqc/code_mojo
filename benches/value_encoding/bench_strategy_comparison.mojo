"""
Comprehensive benchmark suite comparing all Mojo strategies and Qiskit.
Strategies: scalar, simd_v2, fused_v3, v_grid, v_grid_fused, qiskit.
Circuits: prep and prep+iqft (from value encoding).
"""

from butterfly.utils.benchmark_runner import create_runner, should_autosave
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
    var value = 4.7

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
    alias NAME = "comprehensive_strategies"
    alias DESCRIPTION = "Value Encoding: Mojo vs Qiskit "

    var p_cols = List[String]("n", "circuit")
    var b_cols = List[String](
        "scalar",
        "simd_v2",
        "fused_v3",
        "v_grid",
        "v_grid_fused",
        "grid_fusion",
        "q_aer_raw",
        "q_aer_opt",
        "q_latency",
    )

    var runner = create_runner(NAME, DESCRIPTION, p_cols, b_cols, 6)

    var n_values = List[Int]()
    for n in range(
        20, 27
    ):  # Start at n=12 to avoid grid_fusion crashes at small n
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

            try:
                if n <= 22:
                    # Mojo-to-Mojo parity
                    runner.log_progress("Checking Mojo-to-Mojo parity...")
                    try:
                        runner.verify(
                            input,
                            execute_simd_v2_circuit,
                            execute_scalar_circuit,
                            compare_quantum_states,
                            name1="simd_v2",
                            name2="scalar",
                        )
                    except e:
                        runner.log_progress(
                            "!! simd_v2 verification failed: " + String(e)
                        )

                    try:
                        runner.verify(
                            input,
                            execute_fused_v3_circuit,
                            execute_scalar_circuit,
                            compare_quantum_states,
                            name1="fused_v3",
                            name2="scalar",
                        )
                    except e:
                        runner.log_progress(
                            "!! fused_v3 verification failed: " + String(e)
                        )

                    try:
                        runner.verify(
                            input,
                            execute_v_grid_circuit,
                            execute_scalar_circuit,
                            compare_quantum_states,
                            name1="v_grid",
                            name2="scalar",
                        )
                    except e:
                        runner.log_progress(
                            "!! v_grid verification failed: " + String(e)
                        )

                    try:
                        runner.verify(
                            input,
                            execute_v_grid_fused_circuit,
                            execute_scalar_circuit,
                            compare_quantum_states,
                            name1="v_grid_fused",
                            name2="scalar",
                        )
                    except e:
                        runner.log_progress(
                            "!! v_grid_fused verification failed: " + String(e)
                        )

                    try:
                        runner.verify(
                            input,
                            execute_grid_fusion_circuit,
                            execute_scalar_circuit,
                            compare_quantum_states,
                            name1="grid_fusion",
                            name2="scalar",
                        )
                    except e:
                        runner.log_progress(
                            "!! grid_fusion verification failed: " + String(e)
                        )

                if n <= 22:
                    # Against Qiskit reference
                    runner.log_progress("\nChecking against Qiskit ...")
                    runner.verify(
                        input,
                        execute_scalar_circuit,
                        execute_qiskit_circuit,
                        compare_quantum_states,
                        name1="scalar",
                        name2="qiskit",
                    )
            except e:
                print(
                    "!! Verification failed for n="
                    + String(n)
                    + ", "
                    + c_name
                    + ": "
                    + String(e)
                )
                # Continue anyway to see if others pass

            # 2. Performance Phase
            runner.add_perf_result(
                params, "scalar", execute_scalar_circuit, input
            )
            runner.add_perf_result(
                params, "simd_v2", execute_simd_v2_circuit, input
            )
            runner.add_perf_result(
                params, "fused_v3", execute_fused_v3_circuit, input
            )
            runner.add_perf_result(
                params, "v_grid", execute_v_grid_circuit, input
            )
            runner.add_perf_result(
                params, "v_grid_fused", execute_v_grid_fused_circuit, input
            )
            runner.add_perf_result(
                params, "grid_fusion", execute_grid_fusion_circuit, input
            )

            # Qiskit time (via interop wrapper)
            from python import Python

            var py_bench = Python.import_module(
                "butterfly.utils.external_benchmarks"
            )

            if i == 0:
                var q_aer_raw = atof(
                    String(
                        py_bench.benchmark_qiskit_prep(
                            n, 4.7, iters=3, mode="none"
                        )
                    )
                )
                runner.add_result(params, "q_aer_raw", q_aer_raw)

                var q_aer_opt = atof(
                    String(
                        py_bench.benchmark_qiskit_prep(
                            n, 4.7, iters=3, mode="cached"
                        )
                    )
                )
                runner.add_result(params, "q_aer_opt", q_aer_opt)

                var q_latency = atof(
                    String(
                        py_bench.benchmark_qiskit_prep(
                            n, 4.7, iters=3, mode="in_loop"
                        )
                    )
                )
                runner.add_result(params, "q_latency", q_latency)
            else:
                var q_aer_raw = atof(
                    String(
                        py_bench.benchmark_qiskit_value_encoding(
                            n, 4.7, iters=3, mode="none"
                        )
                    )
                )
                runner.add_result(params, "q_aer_raw", q_aer_raw)

                var q_aer_opt = atof(
                    String(
                        py_bench.benchmark_qiskit_value_encoding(
                            n, 4.7, iters=3, mode="cached"
                        )
                    )
                )
                runner.add_result(params, "q_aer_opt", q_aer_opt)

                var q_latency = atof(
                    String(
                        py_bench.benchmark_qiskit_value_encoding(
                            n, 4.7, iters=3, mode="in_loop"
                        )
                    )
                )
                runner.add_result(params, "q_latency", q_latency)

    runner.print_table()
    runner.save_csv(NAME, autosave=should_autosave())
