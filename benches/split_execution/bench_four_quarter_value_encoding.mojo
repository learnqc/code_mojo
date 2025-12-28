"""
Benchmark four-quarter split execution vs fused_v3 for value encoding circuits.

Compares the new four-quarter implementation against the fused_v3 baseline.
"""

from butterfly.utils.benchmark_quantum_circuit_execution import (
    create_quantum_circuit_execution_benchmark,
)
from butterfly.utils.benchmark_utils import parse_benchmark_args
from butterfly.core.execute_split_four_quarters_v2 import (
    execute_split_four_quarters_v2_runtime,
)
from butterfly.core.execute_split_four_quarters_batched import (
    execute_split_four_quarters_batched,
)
from butterfly.core.circuit import QuantumCircuit
from butterfly.core.state import QuantumState
from butterfly.algos.value_encoding_circuit import encode_value_circuit
from collections import List


# Execution functions
fn execute_with_fused_v3(circuit: QuantumCircuit) -> QuantumState:
    var c = circuit.copy()
    var state = QuantumState(c.num_qubits)
    c.execute_fused_v3_dynamic(state)
    return state^


fn execute_with_four_quarter(circuit: QuantumCircuit) -> QuantumState:
    """Generic wrapper that uses the runtime version - no dispatch needed!"""
    var n = circuit.num_qubits
    var c = circuit.copy()  # MUST copy - execute_split modifies circuit!
    var state = QuantumState(n)

    try:
        execute_split_four_quarters_v2_runtime(n, state, c)
    except:
        print("Error executing four_quarter for n=" + String(n))

    return state^


fn execute_with_batched(circuit: QuantumCircuit) -> QuantumState:
    """Batched four-quarter executor with parallelization."""
    var c = circuit.copy()
    var state = QuantumState(c.num_qubits)

    try:
        execute_split_four_quarters_batched(state, c)
    except:
        print("Error executing batched for n=" + String(c.num_qubits))

    return state^


# Circuit builder for modified value encoding (loops stop at n-1)
fn build_value_encoding_circuit(n: Int, value: Float64) -> QuantumCircuit:
    """Build value encoding circuit with loops stopping at n-1 to reduce last-qubit involvement.
    """
    from butterfly.algos.qft import iqft
    from math import pi

    var circuit = QuantumCircuit(n)

    # Apply H and P gates only to first n-1 qubits
    for j in range(n - 1):
        circuit.h(j)
    for j in range(n - 1):
        circuit.p(j, 2 * pi / 2 ** (j + 1) * value)

    # IQFT on first n-1 qubits
    var targets = [n - 2 - j for j in range(n - 1)]
    iqft(circuit=circuit, targets=targets, do_swap=False)

    return circuit^


fn main() raises:
    # Parse command-line arguments
    var (benchmark_id, display_name) = parse_benchmark_args(
        "four_quarter_value_encoding",
        "Four-Quarter Value Encoding",
    )

    # Create function lists
    var funcs = List[fn (QuantumCircuit) -> QuantumState]()
    funcs.append(execute_with_fused_v3)
    funcs.append(execute_with_four_quarter)
    funcs.append(execute_with_batched)

    var names = List[String]()
    names.append("fused_v3")
    names.append("four_quarter")
    names.append("batched")

    var descs = List[String]()
    descs.append("Fused V3 (baseline)")
    descs.append("Four-quarter split")
    descs.append("Batched four-quarter (parallel)")

    # Create all test cases (n=3 to n=26, all with value=4.7)
    var tests = List[Tuple[Int, Float64]]()
    for n in range(3, 27):  # n=3 to n=26
        tests.append((n, 4.7))

    # Run single benchmark with all test cases
    create_quantum_circuit_execution_benchmark(
        funcs,
        names,
        descs,
        build_value_encoding_circuit,
        tests,
        benchmark_id,
        display_name + " (n=3-26, swap=False, exclude_last_qubit=True)",
    )
