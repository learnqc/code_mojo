from collections import List
from math import pi

from butterfly.core.quantum_circuit import QuantumCircuit
from butterfly.core.types import FloatType
from butterfly.algos.value_encoding_circuit import iqft_circuit


fn encode_term(
    coeff: FloatType,
    vars: List[Int],
    mut circuit: QuantumCircuit,
    key_size: Int,
    value_size: Int,
):
    for i in range(value_size):
        var denom = FloatType(2 ** i)
        var theta = FloatType(pi) * coeff / denom
        var target = key_size + i
        if len(vars) > 1:
            var controls = List[Int](capacity=len(vars))
            for j in range(len(vars)):
                controls.append(vars[j])
            circuit.mcp(controls, target, theta)
        elif len(vars) > 0:
            circuit.cp(vars[0], target, theta)
        else:
            circuit.p(target, theta)


fn build_polynomial_circuit(
    key_size: Int,
    value_size: Int,
    terms: List[Tuple[FloatType, List[Int]]],
) -> QuantumCircuit:
    var circuit = QuantumCircuit(key_size + value_size)
    for i in range(key_size):
        circuit.h(i)
    for i in range(value_size):
        circuit.h(key_size + i)

    for term in terms:
        encode_term(
            term[0],
            term[1],
            circuit,
            key_size,
            value_size,
        )

    var targets = List[Int](capacity=value_size)
    for j in range(value_size):
        targets.append(key_size + (value_size - 1 - j))
    iqft_circuit(circuit, targets, swap=False)
    return circuit^


fn poly(
    n_key: Int,
    terms: List[Tuple[FloatType, List[Int]]],
    pr: Bool = False,
) -> List[FloatType]:
    var size = 1 << n_key
    var out = List[FloatType](length=size, fill=FloatType(0))
    for k in range(size):
        var p_k = FloatType(0)
        for term in terms:
            var coeff = term[0]
            var vars = term[1].copy()
            var p_k_m = coeff
            for idx in vars:
                var bit = (k >> idx) & 1
                p_k_m *= FloatType(bit)
            p_k += p_k_m
        out[k] = p_k

    if pr:
        print("")
        for k in range(size):
            print(String(k) + " -> " + String(out[k]))

    return out^
