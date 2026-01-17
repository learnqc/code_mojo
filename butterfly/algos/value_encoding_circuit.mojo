from butterfly.core.quantum_circuit import QuantumCircuit, bit_reverse
from butterfly.core.types import FloatType, pi


fn iqft_circuit(mut qc: QuantumCircuit, targets: List[Int], swap: Bool = False):
    for j in reversed(range(len(targets))):
        qc.h(targets[j])
        for k in reversed(range(j)):
            qc.cp(targets[j], targets[k], -pi / 2 ** (j - k))

    if swap:
        bit_reverse(qc, targets)

fn build_iqft_circuit(n: Int, swap: Bool = False) -> QuantumCircuit:
    var qc = QuantumCircuit(n)
    iqft_circuit(qc, [n - 1 - j for j in range(n)], swap=swap)
    return qc^

fn encode_value_circuit(n: Int, v: FloatType) -> QuantumCircuit:
    qc = QuantumCircuit(n)

    for j in range(n):
        qc.h(j)

    for j in range(n):
        # transform(state, j, P(2 * pi / 2 ** (n - j) * v))
        qc.p(j, FloatType(2 * pi / 2 ** (j + 1) * v))
    iqft_circuit(qc, [n - 1 - j for j in range(n)])
    return qc^
