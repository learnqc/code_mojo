from butterfly.core.circuit import QuantumCircuit
from butterfly.utils.visualization import print_state
from butterfly.algos.qft import iqft as iqft_circuit
from butterfly.core.types import pi
from butterfly.core.types import FloatType


fn encode_value_mojo_circuit(
    mut circuit: QuantumCircuit, n: Int, v: FloatType, swap: Bool = False
):
    """
    Adds value encoding gates to the circuit.

    Args:
        circuit: The QuantumCircuit to add gates to.
        n: Number of qubits.
        v: Value to encode.
        swap: Whether to use swap gates in the IQFT.
    """
    for j in range(n):
        circuit.h(j)

    for j in range(n):
        if swap:
            circuit.p(j, 2 * pi / 2 ** (n - j) * v)
        else:
            circuit.p(j, 2 * pi / 2 ** (j + 1) * v)

    targets = [n - 1 - j for j in range(n)]
    if swap:
        targets = [j for j in range(n)]
    iqft_circuit(circuit, targets, do_swap=swap)


fn main() raises:
    alias v = 4.7
    for n in range(26, 27):
        var circuit = QuantumCircuit(n)
        encode_value_mojo_circuit(circuit, n, v, True)
        circuit.execute()
        print("=== N=", n, " ===")
        for i in range(2**n):
            print(circuit.state[i])
