from butterfly.core.quantum_circuit import QuantumCircuit, bit_reverse
from butterfly.core.types import FloatType

from math import pi


fn iqft_circuit(mut QuantumCircuit: QuantumCircuit, targets: List[Int], swap: Bool = False):
    for j in reversed(range(len(targets))):
        QuantumCircuit.h(targets[j])
        for k in reversed(range(j)):
            QuantumCircuit.cp(targets[j], targets[k], -pi / 2 ** (j - k))

    if swap:
        bit_reverse(QuantumCircuit, targets)

fn build_iqft_circuit(n: Int, swap: Bool = False) -> QuantumCircuit:
    var QuantumCircuit = QuantumCircuit(n)
    iqft_circuit(QuantumCircuit, [n - 1 - j for j in range(n)], swap=swap)
    return QuantumCircuit^

fn encode_value_circuit(n: Int, v: FloatType) -> QuantumCircuit:
    QuantumCircuit = QuantumCircuit(n)

    for j in range(n):
        QuantumCircuit.h(j)

    for j in range(n):
        # transform(state, j, P(2 * pi / 2 ** (n - j) * v))
        QuantumCircuit.p(j, 2 * pi / 2 ** (j + 1) * v)
    iqft_circuit(QuantumCircuit, [n - 1 - j for j in range(n)])
    return QuantumCircuit^
