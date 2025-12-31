"""
Demo benchmark showing thread monitoring with BenchmarkRunner.

Compares different quantum circuit execution strategies and shows
their thread usage patterns.
"""

from butterfly.utils.benchmark_runner import create_runner, should_autosave
from butterfly.core.state import QuantumState
from butterfly.core.circuit import QuantumCircuit
from butterfly.core.execute_fused_v3 import execute_fused_v3
from butterfly.core.execute_as_grid import execute_as_grid
from butterfly.algos.value_encoding_circuit import encode_value_circuits_runtime
from collections import Dict, List


fn execute_scalar(circuit: QuantumCircuit) raises -> QuantumState:
    """Execute using scalar strategy (single-threaded)."""
    var n = circuit.num_qubits
    var state = QuantumState(n)

    from butterfly.core.state import transform, c_transform, bit_reverse_state
    from butterfly.core.circuit import (
        GateTransformation,
        SingleControlGateTransformation,
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
        elif t.isa[BitReversalTransformation]():
            bit_reverse_state(state)

    return state^


fn execute_fused(circuit: QuantumCircuit) raises -> QuantumState:
    """Execute using fused_v3 strategy (parallelized)."""
    var n = circuit.num_qubits
    var state = QuantumState(n)

    if n == 10:
        execute_fused_v3[1 << 10](state, circuit)
    elif n == 15:
        execute_fused_v3[1 << 15](state, circuit)
    elif n == 20:
        execute_fused_v3[1 << 20](state, circuit)

    return state^


fn execute_grid(circuit: QuantumCircuit) raises -> QuantumState:
    """Execute using grid strategy (parallelized)."""
    var n = circuit.num_qubits
    var state = QuantumState(n)
    var col_bits = n - 3
    execute_as_grid(state, circuit, col_bits)
    return state^


fn main() raises:
    print("Thread Monitoring Demo with BenchmarkRunner")
    print("=" * 80)
    print("Comparing thread usage across execution strategies\n")

    var p_cols = List[String]("n", "circuit")
    var b_cols = List[String]("scalar", "fused_v3", "v_grid")
    var runner = create_runner(
        "thread_demo", "Thread Monitoring Demo", p_cols, b_cols, 0
    )

    # Test with different problem sizes
    var n_values = List[Int](10, 15, 20)

    for i in range(len(n_values)):
        var n = n_values[i]
        print("\n" + "=" * 80)
        print("Testing with n=" + String(n) + " qubits")
        print("=" * 80)

        # Create a simple prep circuit
        var circuits = encode_value_circuits_runtime(n, 3.14, swap=False)
        var input = circuits[0].copy()

        var params = Dict[String, String]()
        params["n"] = String(n)
        params["circuit"] = "prep"

        print("\n[Scalar Strategy - Expected: minimal parallelization]")
        runner.add_perf_result_with_threads(
            params, "scalar", execute_scalar, input, iters=3, thread_samples=5
        )

        print("\n[Fused V3 Strategy - Expected: worker thread pool]")
        runner.add_perf_result_with_threads(
            params, "fused_v3", execute_fused, input, iters=3, thread_samples=5
        )

        print("\n[Virtual Grid Strategy - Expected: worker thread pool]")
        runner.add_perf_result_with_threads(
            params, "v_grid", execute_grid, input, iters=3, thread_samples=5
        )

    print("\n" + "=" * 80)
    runner.print_table()

    print("\n✓ Thread monitoring demo complete!")
    print("\nKey insights:")
    print("- Scalar strategy should show minimal thread usage")
    print("- Parallelized strategies (fused_v3, v_grid) show worker threads")
    print("- Thread counts help identify parallelization overhead")
