from butterfly.core.quantum_circuit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalTransform,
    ClassicalReplacementKind,
    replace_quantum_with_classical,
)
from butterfly.core.circuit import GateTransformation
from butterfly.core.types import pi, FloatType, Amplitude
from butterfly.core.gates import P, X
from butterfly.core.state import QuantumState
from utils.variant import Variant

alias Oracle = Variant[QuantumCircuit, ClassicalTransform]

@always_inline
fn is_bit_not_set(mask: Int, i: Int) -> Bool:
    """Return True if the i-th bit of mask is not set."""
    return (mask >> i) & 1 == 0

fn inversion_0_circuit(n: Int) -> QuantumCircuit:
    """
    Reflection-about-zero circuit.
    Logic: X(all) -> MCP(pi) -> X(all). If n = 1, it is just a Z gate.
    """
    var q = QuantumRegister("q", n)
    var qc = QuantumCircuit(q)

    for i in range(n):
        qc.x(i)

    # multi-controlled Z on all but one qubit
    var controls = List[Int]()
    for i in range(n - 1):
        controls.append(i)

    qc.mcp(List[Int](controls), n - 1, pi)

    for i in range(n):
        qc.x(i)

    return qc^


fn diffuser_circuit(n: Int) -> QuantumCircuit:
    """
    Standard diffusion circuit (inversion-about-the-mean).
    Logic: H(all) -> R0 -> H(all)
    Assumes the standard Grover basis.
    """
    var q = QuantumRegister("q", n)
    var qc = QuantumCircuit(q)

    for i in range(n):
        qc.h(i)

    _ = qc.append_circuit(inversion_0_circuit(n), q)

    for i in range(n):
        qc.h(i)

    return qc^

fn classical_oracle(mut state: QuantumState, targets: List[Int]) raises:
    for i in range(len(targets)):
        state[targets[i]] = -state[targets[i]]

fn phase_oracle_match(n: Int, items: List[Int]) -> QuantumCircuit:
    """
    Create a gate-based phase oracle that matches the provided items.
    For each item, flips the phase of the state if it matches the bit pattern.
    Suitable for running on real quantum hardware.
    """
    var q = QuantumRegister("q", n)
    var qc = QuantumCircuit(q)

    for m_idx in range(len(items)):
        var m = items[m_idx]

        for i in range(n):
            if is_bit_not_set(m, i):
                qc.x(i)

        var controls = List[Int]()
        for i in range(n - 1):
            controls.append(i)

        qc.mcp(List[Int](controls), n - 1, pi)

        for i in range(n):
            if is_bit_not_set(m, i):
                qc.x(i)

    return qc^


# fn diagonal_oracle(n: Int, items: List[Int]) -> QuantumCircuit:
#     """
#     Optimized simulation shortcut for the phase oracle.
#     Directly flips the phase of target indices in the state vector.
#     """
#     var qc = QuantumCircuit(n)
#     qc.diagonal_phase_flip(items.copy(), 0, n)
#     return qc^


fn grover_iterate_circuit(oracle: Variant[QuantumCircuit, ClassicalTransform], num_qubits: Int) -> QuantumCircuit:

    var n = num_qubits  # Always use the provided number of qubits
    var qc = QuantumCircuit(n)

    if oracle.isa[QuantumCircuit]():
        _ = qc.append_circuit(oracle[QuantumCircuit], QuantumRegister("q", n))
    if oracle.isa[ClassicalTransform]():
        var tr = oracle[ClassicalTransform].copy()
        qc.add_classical(tr.name, tr.targets, tr.apply)
    _ = qc.append_circuit(diffuser_circuit(n), QuantumRegister("q", n))

    return qc^


fn grover_iterate_circuit_classical_diffuser(
    oracle: Variant[QuantumCircuit, ClassicalTransform],
    num_qubits: Int,
) raises -> QuantumCircuit:
    var qc = grover_iterate_circuit(oracle, num_qubits)
    _ = replace_quantum_with_classical(
        qc,
        List[ClassicalReplacementKind](
            ClassicalReplacementKind.GROVER_DIFFUSER,
        ),
    )
    return qc^

fn grover_circuit(
    items: List[Int],
    num_qubits: Int,
    iterations: Int,
    use_shortcut: Bool = False,
) -> QuantumCircuit:
    """
    Complete Grover's algorithm iterations.
    If use_shortcut is True, uses an optimized simulation oracle.
    Otherwise, uses a gate-based oracle.
    Note: Does NOT include initial preparation.
    """
    var oracle: Oracle
    if use_shortcut:
        # oracle = diagonal_oracle(num_qubits, items)
        oracle = ClassicalTransform(
            "sign_flip_oracle",
            items,
            classical_oracle,
        )
    else:
        oracle = phase_oracle_match(num_qubits, items)

    # var oracle: QuantumCircuit = phase_oracle_match(num_qubits, items)
    var q = QuantumRegister("q", num_qubits)
    var qc = QuantumCircuit(q)

    append_grover_to_register(qc, q, oracle, iterations)

    return qc^


fn append_grover_to_register(
    mut qc: QuantumCircuit,
    register: QuantumRegister,
    oracle: Oracle,
    iterations: Int,
):
    """
    Append Grover iterations (Oracle + Diffuser) to a specific register.
    """

    step = grover_iterate_circuit(oracle, register.length)

    for _ in range(iterations):
        _ = qc.append_circuit(step, register)


fn append_grover_to_register_classical_diffuser(
    mut qc: QuantumCircuit,
    register: QuantumRegister,
    oracle: Oracle,
    iterations: Int,
) raises:
    """
    Append Grover iterations with the diffuser replaced by a classical transform.
    """
    step = grover_iterate_circuit_classical_diffuser(oracle, register.length)

    for _ in range(iterations):
        _ = qc.append_circuit(step, register)
   
