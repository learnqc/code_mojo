from butterfly.core.circuit import QuantumCircuit
from butterfly.core.types import pi
from butterfly.core.gates import H, P


fn qft(mut circuit: QuantumCircuit, targets: List[Int], do_swap: Bool = True):
    """
    Adds QFT gates to the circuit for the specified targets.

    Args:
        circuit: The QuantumCircuit to add gates to.
        targets: List of target qubit indices.
        do_swap: If True, adds an efficient bit-reversal operation to reverse qubit order.
    """
    if do_swap:
        if len(targets) == circuit.num_qubits:
            circuit.bit_reverse()
        else:
            # Partial swap: swap targets[i] with targets[k-1-i]
            var k = len(targets)
            for i in range(k // 2):
                circuit.swap(targets[i], targets[k - 1 - i])

    for j in range(len(targets)):
        for k in range(j):
            # Controlled phases from previous qubits
            circuit.cp(targets[j], targets[k], pi / (2 ** (j - k)))
        circuit.h(targets[j])


fn iqft(mut circuit: QuantumCircuit, targets: List[Int], do_swap: Bool = True):
    """
    Adds IQFT gates to the circuit for the specified targets.
    Matches the logic in butterfly.core.state.iqft.

    Args:
        circuit: The QuantumCircuit to add gates to.
        targets: List of target qubit indices.
        do_swap: If True, adds an efficient bit-reversal operation to reverse qubit order.
    """
    for j in reversed(range(len(targets))):
        circuit.h(targets[j])
        for k in reversed(range(j)):
            # Note: targets[j] is control, targets[k] is target in state.mojo implementation
            circuit.cp(targets[k], targets[j], -pi / (2 ** (j - k)))

    if do_swap:
        if len(targets) == circuit.num_qubits:
            circuit.bit_reverse()
        else:
            # Partial swap: swap targets[i] with targets[k-1-i]
            var k = len(targets)
            for i in range(k // 2):
                circuit.swap(targets[i], targets[k - 1 - i])
