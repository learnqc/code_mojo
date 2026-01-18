from collections import List

from butterfly.core.circuit import Register
from butterfly.core.quantum_circuit import QuantumCircuit
from butterfly.core.types import Amplitude, Complex
from butterfly.algos.value_encoding_circuit import iqft_circuit


# WARNING: FULL UNITARY MOD-EXP (DEMO ONLY)
# This builds a dense 2^(n_exp+n_value) x 2^(n_exp+n_value) matrix.
# It is exact and reversible, but only practical for very small N.
# Use for visualization and tiny demos (e.g., N=15).


fn gcd(a: Int, b: Int) -> Int:
    var x = a
    var y = b
    while y != 0:
        var tmp = x % y
        x = y
        y = tmp
    return x


fn modexp(base: Int, exp: Int, modulus: Int) -> Int:
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


fn modexp_unitary_matrix(
    n_exp: Int,
    n_value: Int,
    a: Int,
    modulus: Int,
    warn: Bool = True,
) raises -> List[Amplitude]:
    if n_exp <= 0 or n_value <= 0:
        raise Error("modexp_unitary_matrix expects positive register sizes")
    if modulus <= 1:
        raise Error("modexp_unitary_matrix expects modulus > 1")
    if gcd(a, modulus) != 1:
        raise Error("modexp_unitary_matrix expects a and modulus coprime")
    var total = n_exp + n_value
    var dim = 1 << total
    if warn:
        print(
            "WARNING: building full unitary for mod-exp on "
            + String(total)
            + " qubits (dim="
            + String(dim)
            + "). Demo only."
        )
    if dim > (1 << 12):
        raise Error("modexp_unitary_matrix too large for demo")

    var u = List[Amplitude](length=dim * dim, fill=Complex(0, 0))
    var x_mask = (1 << n_exp) - 1
    var y_mask = (1 << n_value) - 1
    for idx in range(dim):
        var x = idx & x_mask
        var y = (idx >> n_exp) & y_mask
        var y_out = y
        if y < modulus:
            var ax = modexp(a, x, modulus)
            y_out = (y * ax) % modulus
        var out_idx = x | (y_out << n_exp)
        u[out_idx * dim + idx] = Complex(1, 0)
    return u^


fn modexp_unitary_circuit(
    n_exp: Int,
    n_value: Int,
    a: Int,
    modulus: Int,
    warn: Bool = True,
) raises -> QuantumCircuit:
    var total = n_exp + n_value
    var qc = QuantumCircuit(total)
    var exp = Register("exp", n_exp)
    var value = Register("value", n_value, n_exp)

    for i in range(exp.length):
        qc.h(exp[i])
    qc.x(value[0])

    var u = modexp_unitary_matrix(n_exp, n_value, a, modulus, warn=warn)
    qc.append_u(u, Register("all", total))
    return qc^


fn order_finding_unitary_circuit(
    n_exp: Int,
    n_value: Int,
    a: Int,
    modulus: Int,
    swap: Bool = False,
    warn: Bool = True,
) raises -> QuantumCircuit:
    var qc = modexp_unitary_circuit(n_exp, n_value, a, modulus, warn=warn)
    var iqft = QuantumCircuit(n_exp)
    _ = iqft_circuit(iqft, [n_exp - 1 - j for j in range(n_exp)], swap=swap)
    _ = qc.append_circuit(iqft, Register("exp", n_exp))
    return qc^
