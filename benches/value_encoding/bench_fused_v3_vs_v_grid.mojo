"""
Benchmark fused_v3 vs Virtual Grid (V5) with verification.

Compares:
- fused_v3: Gate fusion with arithmetic reduction
- v_grid: Virtual Grid V5 with cache locality

Only benchmarks prep and iqft circuits (bit reversal is strategy-agnostic).
"""

from butterfly.utils.benchmark_runner import create_runner, should_autosave
from butterfly.core.state import QuantumState
from butterfly.core.circuit import QuantumCircuit
from butterfly.core.execute_as_grid import execute_as_grid
from butterfly.core.execute_fused_v3 import execute_fused_v3
from butterfly.algos.value_encoding_circuit import encode_value_circuits_runtime
from butterfly.utils.visualization import print_state
from collections import Dict, List


# --- Execution Functions (all take QuantumCircuit) ---


fn execute_fused_v3_circuit(circuit: QuantumCircuit) raises -> QuantumState:
    """Execute using fused_v3 with compile-time N dispatch."""
    var n = circuit.num_qubits
    var state = QuantumState(n)

    # Compile-time dispatch for N
    if n == 3:
        execute_fused_v3[1 << 3](state, circuit)
    if n == 15:
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

    return state^


fn execute_v_grid_circuit(circuit: QuantumCircuit) raises -> QuantumState:
    """Execute using Virtual Grid V5."""
    var n = circuit.num_qubits
    var state = QuantumState(n)
    var col_bits = n - 3
    try:
        execute_as_grid(state, circuit, col_bits)
    except:
        print("Execution failed")

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
        print_state(val1)
        print_state(val2)
        raise Error(
            "Verification failed: QuantumStates differ by " + String(diff_sum)
        )


fn main() raises:
    alias NAME = "bench_fused_v3_vs_v_grid"
    alias DESCRIPTION = "Fused V3 vs Virtual Grid V5 (Verified)"

    # Define columns
    var param_cols = List[String]("n", "circuit")
    var bench_cols = List[String]("fused_v3_ms", "v_grid_ms")

    # Create runner
    var runner = create_runner(NAME, DESCRIPTION, param_cols, bench_cols)

    print("Benchmarking Fused V3 vs Virtual Grid V5 (with verification)")
    print("=" * 80)
    print()

    var n_values = List[Int](3, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25)
    var circuit_names = List[String]("prep", "iqft")

    # Verification phase
    print("Verification Phase")
    print("-" * 80)
    for i in range(len(n_values)):
        var n = n_values[i]
        var circuits = encode_value_circuits_runtime(n, 4.7)
        for circuit_idx in range(2):  # Only prep and iqft
            var input = circuits[circuit_idx].copy()

            var name1 = "fused_v3"
            var name2 = "v_grid"
            runner.log_progress(
                "\nVerifying n="
                + String(n)
                + ", circuit="
                + circuit_names[circuit_idx]
                + ", "
                + name1
                + " vs "
                + name2
                + " ..."
            )
            runner.verify[QuantumCircuit, QuantumState](
                input,
                execute_fused_v3_circuit,
                execute_v_grid_circuit,
                compare_quantum_states,
                name1=name1,
                name2=name2,
                tolerance=1e-10,
            )

    print("\n✓ All verifications passed!")
    print()

    # Performance benchmarking phase
    print("Performance Benchmarking Phase")
    print("-" * 80)
    for i in range(len(n_values)):
        var n = n_values[i]
        var circuits = encode_value_circuits_runtime(n, 4.7)
        for circuit_idx in range(2):  # Only prep and iqft
            var input = circuits[circuit_idx].copy()
            var params = Dict[String, String]()
            params["n"] = String(n)
            params["circuit"] = circuit_names[circuit_idx]

            runner.log_progress(
                "\nBenchmarking n="
                + String(n)
                + ", circuit="
                + circuit_names[circuit_idx]
                + " ..."
            )

            runner.add_perf_result(
                params, "fused_v3_ms", execute_fused_v3_circuit, input
            )
            runner.add_perf_result(
                params, "v_grid_ms", execute_v_grid_circuit, input
            )

    print()
    runner.print_table(show_winner=True)
    runner.save_csv(NAME, autosave=should_autosave())

    print("\nStrategies:")
    print("  fused_v3_ms = Gate fusion with arithmetic reduction")
    print("  v_grid_ms   = Virtual Grid V5 (cache locality)")
    print("\nCircuits:")
    print("  prep = Preparation phase (Circuit 0)")
    print("  iqft = IQFT phase (Circuit 1)")
    print("\nNote: Bit reversal skipped (strategy-agnostic)")
