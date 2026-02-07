from collections import List

from butterfly.algos.function_encoding import build_polynomial_circuit
from butterfly.algos.value_encoding_circuit import iqft_circuit
from butterfly.core.quantum_circuit import QuantumCircuit, QuantumRegister
from butterfly.core.types import FloatType


fn modexp_int(base: Int, exp: Int, modulus: Int) -> Int:
    if modulus <= 1:
        return 0
    var result = 1 % modulus
    var b = base % modulus
    var e = exp
    while e > 0:
        if (e & 1) == 1:
            result = (result * b) % modulus
        b = (b * b) % modulus
        e >>= 1
    return result


fn build_modexp_terms(
    n_exp: Int,
    modulus: Int,
    base: Int = 2,
) -> List[Tuple[FloatType, List[Int]]]:
    var total = 1 << n_exp
    var coeffs = List[FloatType](length=total, fill=FloatType(0))
    for x in range(total):
        coeffs[x] = FloatType(modexp_int(base, x, modulus))
    for i in range(n_exp):
        for mask in range(total):
            if ((mask >> i) & 1) == 1:
                coeffs[mask] -= coeffs[mask ^ (1 << i)]
    var terms = List[Tuple[FloatType, List[Int]]]()
    for mask in range(total):
        var vars = List[Int]()
        for i in range(n_exp):
            if ((mask >> i) & 1) == 1:
                vars.append(i)
        terms.append((coeffs[mask], vars^))
    return terms^


fn build_shor_polynomial_circuit(
    n_exp: Int,
    n_value: Int,
    modulus: Int,
    base: Int = 2,
) raises -> QuantumCircuit:
    var terms = build_modexp_terms(n_exp, modulus, base)
    var qc = build_polynomial_circuit(n_exp, n_value, terms)
    var iqft = QuantumCircuit(n_exp)
    _ = iqft_circuit(iqft, [n_exp - 1 - j for j in range(n_exp)], swap=False)
    _ = qc.append_circuit(iqft, QuantumRegister("exp", n_exp))
    return qc^
