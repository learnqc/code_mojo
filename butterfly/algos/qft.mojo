from butterfly.core.circuit import QuantumCircuit
from butterfly.core.types import pi
from butterfly.core.gates import H, P


fn qft(mut circuit: QuantumCircuit, targets: List[Int], do_swap: Bool = True):
    """
    Adds QFT gates to the circuit for the specified targets.

    This is a thin wrapper around the core _qft implementation.

    Args:
        circuit: The QuantumCircuit to add gates to.
        targets: List of target qubit indices.
        do_swap: If True, adds an efficient bit-reversal operation to reverse qubit order.
    """
    from butterfly.core.circuit import _qft

    _qft(circuit, targets, do_swap)


fn iqft(mut circuit: QuantumCircuit, targets: List[Int], do_swap: Bool = True):
    """
    Adds IQFT gates to the circuit for the specified targets.

    This is a thin wrapper around the core _iqft implementation.

    Args:
        circuit: The QuantumCircuit to add gates to.
        targets: List of target qubit indices.
        do_swap: If True, adds an efficient bit-reversal operation to reverse qubit order.
    """
    from butterfly.core.circuit import _iqft

    _iqft(circuit, targets, do_swap)
