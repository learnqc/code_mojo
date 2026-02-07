from butterfly.core.quantum_circuit import (
    QuantumCircuit,
    QuantumTransformation,
    ClassicalReplacementKind,
    replace_quantum_with_classical,
)
from butterfly.core.circuit import (
    SwapTransformation,
    QubitReversalTransformation,
    ClassicalTransformation,
)
from butterfly.core.state import QuantumState
from butterfly.core.executors import execute
from butterfly.utils.context import ExecContext


fn assert_is_classical(
    tr: QuantumTransformation,
    expected_name: String,
) raises:
    if not tr.isa[ClassicalTransformation[QuantumState]]():
        raise Error("Expected classical transformation: " + expected_name)
    var cl = tr[ClassicalTransformation[QuantumState]].copy()
    if cl.name != expected_name:
        raise Error(
            "Expected classical name "
            + expected_name
            + ", got "
            + cl.name
        )


fn test_swap_and_qrev_replacements() raises:
    var circuit = QuantumCircuit(3)
    circuit.swap(0, 2)
    circuit.qubit_reversal(List[Int](0, 1, 2))

    var replaced = replace_quantum_with_classical(
        circuit,
        List[ClassicalReplacementKind](
            ClassicalReplacementKind.SWAP,
            ClassicalReplacementKind.QUBIT_REVERSAL,
        ),
    )
    if replaced != 2:
        raise Error("Expected 2 replacements, got " + String(replaced))

    var t0 = circuit.transformations[0]
    var t1 = circuit.transformations[1]
    if t0.isa[SwapTransformation]():
        raise Error("Swap was not replaced")
    if t1.isa[QubitReversalTransformation]():
        raise Error("Qubit reversal was not replaced")
    assert_is_classical(t0, "SWAP")
    assert_is_classical(t1, "BITREV")


fn test_selective_replacement() raises:
    var circuit = QuantumCircuit(3)
    circuit.swap(0, 2)
    circuit.qubit_reversal(List[Int](0, 1, 2))

    var replaced = replace_quantum_with_classical(
        circuit,
        List[ClassicalReplacementKind](ClassicalReplacementKind.SWAP),
    )
    if replaced != 1:
        raise Error("Expected 1 replacement, got " + String(replaced))

    var t0 = circuit.transformations[0]
    var t1 = circuit.transformations[1]
    assert_is_classical(t0, "SWAP")
    if not t1.isa[QubitReversalTransformation]():
        raise Error("Qubit reversal should remain quantum")


fn test_execute_after_replacement() raises:
    var circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.swap(0, 1)
    circuit.h(1)

    _ = replace_quantum_with_classical(
        circuit,
        List[ClassicalReplacementKind](ClassicalReplacementKind.SWAP),
    )

    var state = QuantumState(2)
    execute(state, circuit, ExecContext())


fn main() raises:
    test_swap_and_qrev_replacements()
    test_selective_replacement()
    test_execute_after_replacement()
