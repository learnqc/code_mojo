"""
Benchmark GridQuantumState gate performance across different qubit positions.

Tests all 8 gates (X, Y, Z, H, P, RX, RY, RZ) on three positions:
- First qubit (0)
- Middle qubit (n//2)
- Last qubit (n-1)

This reveals how grid strategies perform when targeting:
- Column qubits (fast path for grid)
- Row qubits (cross-row coupling for grid)

Compares:
- Generic: Standard execute()
- SIMD v2: execute_simd_v2_dynamic()
- Grid 2-row: GridQuantumState with row_bits=1
- Grid 4-row: GridQuantumState with row_bits=2
- Grid 8-row: GridQuantumState with row_bits=3
"""
# pixi run mojo run -I . benches/value_encoding/bench_grid_gate_positions.mojo

from butterfly.utils.benchmark_runner import create_runner
from butterfly.core.grid_state_old import GridQuantumState
from butterfly.core.state import QuantumState
from butterfly.core.circuit import QuantumCircuit
from collections import Dict, List
from math import pi


# --- Execution Functions (all take Tuple[Int, String, Int]) ---
# Tuple = (n, gate_name, target_qubit)


fn execute_generic(params: Tuple[Int, String, Int]) raises -> QuantumState:
    """Execute using standard execute()."""
    var n = params[0]
    var gate_name = params[1]
    var target = params[2]

    var circuit = QuantumCircuit(n)

    # Apply the specified gate
    if gate_name == "X":
        circuit.x(target)
    elif gate_name == "Y":
        circuit.y(target)
    elif gate_name == "Z":
        circuit.z(target)
    elif gate_name == "H":
        circuit.h(target)
    elif gate_name == "P":
        circuit.p(target, pi / 4)
    elif gate_name == "RX":
        circuit.rx(target, pi / 4)
    elif gate_name == "RY":
        circuit.ry(target, pi / 4)
    elif gate_name == "RZ":
        circuit.rz(target, pi / 4)

    var state = QuantumState(n)
    circuit.execute(state)
    return state^


fn execute_simd_v2(params: Tuple[Int, String, Int]) raises -> QuantumState:
    """Execute using SIMD v2."""
    var n = params[0]
    var gate_name = params[1]
    var target = params[2]

    var circuit = QuantumCircuit(n)

    if gate_name == "X":
        circuit.x(target)
    elif gate_name == "Y":
        circuit.y(target)
    elif gate_name == "Z":
        circuit.z(target)
    elif gate_name == "H":
        circuit.h(target)
    elif gate_name == "P":
        circuit.p(target, pi / 4)
    elif gate_name == "RX":
        circuit.rx(target, pi / 4)
    elif gate_name == "RY":
        circuit.ry(target, pi / 4)
    elif gate_name == "RZ":
        circuit.rz(target, pi / 4)

    var state = QuantumState(n)
    circuit.execute_simd_v2_dynamic(state)
    return state^


fn execute_grid_2row(params: Tuple[Int, String, Int]) raises -> QuantumState:
    """Execute using GridQuantumState with 2 rows."""
    var n = params[0]
    var gate_name = params[1]
    var target = params[2]

    var circuit = QuantumCircuit(n)

    if gate_name == "X":
        circuit.x(target)
    elif gate_name == "Y":
        circuit.y(target)
    elif gate_name == "Z":
        circuit.z(target)
    elif gate_name == "H":
        circuit.h(target)
    elif gate_name == "P":
        circuit.p(target, pi / 4)
    elif gate_name == "RX":
        circuit.rx(target, pi / 4)
    elif gate_name == "RY":
        circuit.ry(target, pi / 4)
    elif gate_name == "RZ":
        circuit.rz(target, pi / 4)

    var grid = GridQuantumState(n, 1)

    # Dispatch based on n for compile-time row_size
    if n == 10:
        grid.execute[1 << 9](circuit)
    elif n == 15:
        grid.execute[1 << 14](circuit)
    elif n == 20:
        grid.execute[1 << 19](circuit)
    elif n == 25:
        grid.execute[1 << 24](circuit)

    # Convert to QuantumState
    var state = QuantumState(n)
    for i in range(grid.size()):
        state.re[i] = grid.re[i]
        state.im[i] = grid.im[i]
    return state^


fn execute_grid_4row(params: Tuple[Int, String, Int]) raises -> QuantumState:
    """Execute using GridQuantumState with 4 rows."""
    var n = params[0]
    var gate_name = params[1]
    var target = params[2]

    var circuit = QuantumCircuit(n)

    if gate_name == "X":
        circuit.x(target)
    elif gate_name == "Y":
        circuit.y(target)
    elif gate_name == "Z":
        circuit.z(target)
    elif gate_name == "H":
        circuit.h(target)
    elif gate_name == "P":
        circuit.p(target, pi / 4)
    elif gate_name == "RX":
        circuit.rx(target, pi / 4)
    elif gate_name == "RY":
        circuit.ry(target, pi / 4)
    elif gate_name == "RZ":
        circuit.rz(target, pi / 4)

    var grid = GridQuantumState(n, 2)

    if n == 10:
        grid.execute[1 << 8](circuit)
    elif n == 15:
        grid.execute[1 << 13](circuit)
    elif n == 20:
        grid.execute[1 << 18](circuit)
    elif n == 25:
        grid.execute[1 << 23](circuit)

    # Convert to QuantumState
    var state = QuantumState(n)
    for i in range(grid.size()):
        state.re[i] = grid.re[i]
        state.im[i] = grid.im[i]
    return state^


fn execute_grid_8row(params: Tuple[Int, String, Int]) raises -> QuantumState:
    """Execute using GridQuantumState with 8 rows."""
    var n = params[0]
    var gate_name = params[1]
    var target = params[2]

    var circuit = QuantumCircuit(n)

    if gate_name == "X":
        circuit.x(target)
    elif gate_name == "Y":
        circuit.y(target)
    elif gate_name == "Z":
        circuit.z(target)
    elif gate_name == "H":
        circuit.h(target)
    elif gate_name == "P":
        circuit.p(target, pi / 4)
    elif gate_name == "RX":
        circuit.rx(target, pi / 4)
    elif gate_name == "RY":
        circuit.ry(target, pi / 4)
    elif gate_name == "RZ":
        circuit.rz(target, pi / 4)

    var grid = GridQuantumState(n, 3)

    if n == 10:
        grid.execute[1 << 7](circuit)
    elif n == 15:
        grid.execute[1 << 12](circuit)
    elif n == 20:
        grid.execute[1 << 17](circuit)
    elif n == 25:
        grid.execute[1 << 22](circuit)

    # Convert to QuantumState
    var state = QuantumState(n)
    for i in range(grid.size()):
        state.re[i] = grid.re[i]
        state.im[i] = grid.im[i]
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
    alias NAME = "grid_gate_positions"
    alias DESCRIPTION = "Grid Gate Performance by Position"

    # Define columns
    var param_cols = List[String]("n", "gate", "position")
    var bench_cols = List[String](
        "generic_ms",
        "simd_v2_ms",
        "grid_2row_ms",
        "grid_4row_ms",
        "grid_8row_ms",
    )

    # Create runner
    var runner = create_runner(NAME, DESCRIPTION, param_cols, bench_cols)

    print("Benchmarking Grid Gate Performance by Position")
    print("=" * 80)
    print()

    var n_values = List[Int](10, 15, 20, 25)
    var gates = List[String]("X", "Y", "Z", "H", "P", "RX", "RY", "RZ")

    # Verification phase
    print("Verification Phase")
    print("-" * 80)

    for i in range(len(n_values)):
        var n = n_values[i]
        var middle = n // 2
        var last = n - 1

        print("Verifying n=" + String(n) + "...")

        for j in range(len(gates)):
            var gate = gates[j]

            # Test first, middle, last positions
            for pos_idx in range(3):
                var target = 0
                if pos_idx == 0:
                    target = 0
                elif pos_idx == 1:
                    target = middle
                else:
                    target = last

                var input = (n, gate, target)

                # Verify all strategies against generic baseline
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
                    execute_grid_2row,
                    compare_quantum_states,
                    name1="generic",
                    name2="grid_2row",
                    tolerance=1e-10,
                )

                runner.verify(
                    input,
                    execute_generic,
                    execute_grid_4row,
                    compare_quantum_states,
                    name1="generic",
                    name2="grid_4row",
                    tolerance=1e-10,
                )

                runner.verify(
                    input,
                    execute_generic,
                    execute_grid_8row,
                    compare_quantum_states,
                    name1="generic",
                    name2="grid_8row",
                    tolerance=1e-10,
                )

    print("✓ All verifications passed!")
    print()

    # Performance benchmarking phase
    print("Performance Benchmarking Phase")
    print("-" * 80)

    for i in range(len(n_values)):
        var n = n_values[i]
        var middle = n // 2
        var last = n - 1

        for j in range(len(gates)):
            var gate = gates[j]

            # Test first, middle, last positions
            for pos_idx in range(3):
                var target = 0
                var pos_name = ""

                if pos_idx == 0:
                    target = 0
                    pos_name = "first"
                elif pos_idx == 1:
                    target = middle
                    pos_name = "middle"
                else:
                    target = last
                    pos_name = "last"

                var input = (n, gate, target)
                var params = Dict[String, String]()
                params["n"] = String(n)
                params["gate"] = gate
                params["position"] = pos_name

                runner.log_progress(
                    "n=" + String(n) + ", gate=" + gate + ", pos=" + pos_name
                )

                runner.add_perf_result(
                    params, "generic_ms", execute_generic, input
                )
                runner.add_perf_result(
                    params, "simd_v2_ms", execute_simd_v2, input
                )
                runner.add_perf_result(
                    params, "grid_2row_ms", execute_grid_2row, input
                )
                runner.add_perf_result(
                    params, "grid_4row_ms", execute_grid_4row, input
                )
                runner.add_perf_result(
                    params, "grid_8row_ms", execute_grid_8row, input
                )

    print()
    runner.print_table(show_winner=True)
    runner.save_csv(NAME)

    print("\nStrategies:")
    print("  generic_ms    = Standard execute()")
    print("  simd_v2_ms    = SIMD v2 dynamic")
    print("  grid_2row_ms  = GridQuantumState with 2 rows")
    print("  grid_4row_ms  = GridQuantumState with 4 rows")
    print("  grid_8row_ms  = GridQuantumState with 8 rows")
    print("\nResults saved with timestamp:", String(runner.timestamp))
