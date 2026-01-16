from butterfly.core.quantum_circuit import QuantumCircuit, QuantumRegister
from butterfly.core.types import pi, FloatType, Amplitude
from butterfly.core.gates import P, X

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


fn grover_iterate_circuit(oracle: QuantumCircuit) -> QuantumCircuit:
    """
    Single Grover iteration: Oracle -> Standard Diffuser.
    """
    var n = oracle.num_qubits
    var q = QuantumRegister("q", n)
    var qc = QuantumCircuit(q)

    # oracle
    _ =qc.append_circuit(oracle, q)

    # diffusion
    _ = qc.append_circuit(diffuser_circuit(n), q)

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
    var oracle: QuantumCircuit
    if use_shortcut:
        # oracle = diagonal_oracle(num_qubits, items)
        oracle = phase_oracle_match(num_qubits, items)
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
    oracle: QuantumCircuit,
    iterations: Int,
):
    """
    Append Grover iterations (Oracle + Diffuser) to a specific register.
    """
    var step = grover_iterate_circuit(oracle)
    for _ in range(iterations):
        _ = qc.append_circuit(step, register)
