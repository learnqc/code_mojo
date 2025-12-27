"""
Benchmark value encoding using the new create_benchmark utility.

This demonstrates the simplified approach where you just define functions
and test cases, then call create_benchmark() to handle everything.
"""

from butterfly.utils.benchmark_quantum_circuit_execution import (
    create_quantum_circuit_execution_benchmark,
)
from butterfly.utils.benchmark_utils import parse_benchmark_args
from butterfly.core.circuit import QuantumCircuit
from butterfly.core.state import QuantumState
from butterfly.algos.value_encoding_circuit import encode_value_circuit
from butterfly.core.execution_strategy import (
    EXECUTION_STRATEGIES,
    GENERIC,
    SIMD_STRATEGY,
    SIMD_V2,
    FUSED_V3,
    get_strategy_name,
    get_strategy_description,
)
from butterfly.core.grid_state import GridQuantumState
from butterfly.core.grid_state_hybrid import HybridGrid
from butterfly.core.types import FloatType
from collections import List


# Define execution functions for each strategy
fn execute_with_generic(circuit: QuantumCircuit) -> QuantumState:
    var c = circuit.copy()
    return c.run_with_strategy(GENERIC)


fn execute_with_simd(circuit: QuantumCircuit) -> QuantumState:
    var c = circuit.copy()
    return c.run_with_strategy(SIMD_STRATEGY)


fn execute_with_simd_v2(circuit: QuantumCircuit) -> QuantumState:
    var c = circuit.copy()
    return c.run_with_strategy(SIMD_V2)


fn execute_with_fused_v3(circuit: QuantumCircuit) -> QuantumState:
    var c = circuit.copy()
    return c.run_with_strategy(FUSED_V3)


# Grid execution functions
fn execute_with_grid(circuit: QuantumCircuit) -> QuantumState:
    alias row_bits = 1  # 2 rows
    var grid = GridQuantumState(circuit.num_qubits, row_bits)
    try:
        grid.execute[1 << row_bits](circuit)
    except:
        pass  # Should not happen
    # Convert to QuantumState for compatibility
    var row_re = List[FloatType]()
    var row_im = List[FloatType]()
    for r in range(grid.num_rows):
        for c in range(grid.row_size):
            row_re.append(grid.re[grid.get_row_offset(r) + c])
            row_im.append(grid.im[grid.get_row_offset(r) + c])
    return QuantumState(row_re^, row_im^)


fn execute_with_hybrid(circuit: QuantumCircuit) -> QuantumState:
    alias row_bits = 1  # 2 rows
    var hybrid = HybridGrid(circuit.num_qubits, row_bits)
    try:
        hybrid.execute[True](circuit)
    except:
        pass  # Should not happen
    # Convert to QuantumState for compatibility
    var row_re = List[FloatType]()
    var row_im = List[FloatType]()
    for r in range(hybrid.num_rows):
        for c in range(hybrid.row_size):
            row_re.append(hybrid.re[r * hybrid.row_size + c])
            row_im.append(hybrid.im[r * hybrid.row_size + c])
    return QuantumState(row_re^, row_im^)


# Circuit builder function
fn build_value_encoding_circuit(n: Int, value: Float64) -> QuantumCircuit:
    var circuit = QuantumCircuit(n)
    encode_value_circuit(n, circuit, value)
    return circuit^


fn main() raises:
    # Parse command-line arguments (allows runner to override from JSON)
    var (benchmark_id, display_name) = parse_benchmark_args(
        "value_encoding_strategies",
        "Value Encoding Strategies",
    )

    # Create function list
    var functions = List[fn (QuantumCircuit) -> QuantumState]()
    functions.append(execute_with_generic)
    functions.append(execute_with_simd)
    functions.append(execute_with_simd_v2)
    functions.append(execute_with_fused_v3)
    functions.append(execute_with_grid)
    functions.append(execute_with_hybrid)

    # Create name list
    var names = List[String]()
    var strategies = materialize[EXECUTION_STRATEGIES]()
    for i in range(len(strategies)):
        names.append(get_strategy_name(strategies[i]))
    names.append("execute_grid")
    names.append("execute_hybrid")

    # Create description list
    var descriptions = List[String]()
    for i in range(len(strategies)):
        descriptions.append(get_strategy_description(strategies[i]))
    descriptions.append("Grid state (2 rows)")
    descriptions.append("Hybrid grid (2 rows)")

    # Define test cases
    var test_cases = List[Tuple[Int, Float64]]()
    test_cases.append((10, 42.0))
    test_cases.append((12, 123.0))
    test_cases.append((15, 456.0))
    test_cases.append((18, 789.0))

    # One function call does everything!
    # - Automatic date-based path organization
    # - Verification
    # - Benchmarking
    # - Table printing
    # - CSV export
    # - Description display
    create_quantum_circuit_execution_benchmark(
        functions,
        names,
        descriptions,
        build_value_encoding_circuit,
        test_cases,
        benchmark_id,  # From parse_benchmark_args (can be overridden by runner)
        display_name,  # From parse_benchmark_args (can be overridden by runner)
    )
