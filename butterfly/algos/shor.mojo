from collections import List

from butterfly.core.quantum_circuit import QuantumCircuit, QuantumRegister
from butterfly.core.state import QuantumState
from butterfly.core.executors import execute
from butterfly.utils.context import ExecContext
from butterfly.core.types import FloatType
from butterfly.algos.value_encoding_circuit import iqft_circuit


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


fn modpow(base: Int, exp: Int, modulus: Int) -> Int:
    return modexp(base, exp, modulus)


fn apply_modexp(
    mut state: QuantumState,
    targets: List[Int],
) raises:
    if len(targets) < 4:
        raise Error("MOD_EXP expects 4 parameters: n_exp, n_value, a, N")
    var n_exp = targets[0]
    var n_value = targets[1]
    var a = targets[2]
    var modulus = targets[3]
    if n_exp <= 0 or n_value <= 0:
        raise Error("MOD_EXP expects positive register sizes")
    if modulus <= 1:
        raise Error("MOD_EXP expects modulus > 1")
    if gcd(a, modulus) != 1:
        raise Error("MOD_EXP expects a and modulus to be coprime")

    var total = n_exp + n_value
    var size = state.size()
    if size != (1 << total):
        raise Error("State size mismatch for MOD_EXP")

    var out_re = List[FloatType](length=size, fill=0.0)
    var out_im = List[FloatType](length=size, fill=0.0)
    var x_mask = (1 << n_exp) - 1
    var y_mask = (1 << n_value) - 1

    for idx in range(size):
        var x = idx & x_mask
        var y = (idx >> n_exp) & y_mask
        var y_out = y
        if y < modulus:
            var ax = modexp(a, x, modulus)
            y_out = (y * ax) % modulus
        var out_idx = x | (y_out << n_exp)
        out_re[out_idx] = state.re[idx]
        out_im[out_idx] = state.im[idx]

    state.re = out_re^
    state.im = out_im^
    state.invalidate_buffers()


fn exponent_marginal_probs(
    state: QuantumState,
    n_exp: Int,
    n_value: Int,
) -> List[FloatType]:
    var total_bits = n_exp + n_value
    var size = state.size()
    if total_bits <= 0 or size != (1 << total_bits):
        return List[FloatType]()
    var exp_size = 1 << n_exp
    var probs = List[FloatType](length=exp_size, fill=0.0)
    var mask = exp_size - 1
    for idx in range(size):
        var x = idx & mask
        var amp = state[idx]
        var prob = amp.re * amp.re + amp.im * amp.im
        probs[x] += prob
    return probs^


fn continued_fraction_best(
    numer: Int,
    denom: Int,
    max_den: Int,
) -> Tuple[Int, Int]:
    if denom == 0:
        return (0, 1)
    var n = numer
    var d = denom
    var cf = List[Int]()
    while d != 0:
        var q = n // d
        cf.append(q)
        var r = n - q * d
        n = d
        d = r

    var p0 = 1
    var q0 = 0
    var p1 = cf[0]
    var q1 = 1
    var best_p = p1
    var best_q = q1
    for i in range(1, len(cf)):
        var p = cf[i] * p1 + p0
        var q = cf[i] * q1 + q0
        if q > max_den:
            break
        best_p = p
        best_q = q
        p0 = p1
        q0 = q1
        p1 = p
        q1 = q
    return (best_p, best_q)


fn estimate_order_from_measurement(
    measured: Int,
    n_exp: Int,
    a: Int,
    modulus: Int,
    max_den: Int = 0,
) -> Optional[Int]:
    if n_exp <= 0:
        return None
    var denom = 1 << n_exp
    var max_r = max_den if max_den > 0 else modulus
    var (p, q) = continued_fraction_best(measured, denom, max_r)
    if q <= 0:
        return None
    if modpow(a, q, modulus) == 1:
        return q
    for k in range(2, 5):
        var cand = q * k
        if cand > max_r:
            break
        if modpow(a, cand, modulus) == 1:
            return cand
    return None


fn estimate_order_from_state(
    state: QuantumState,
    n_exp: Int,
    n_value: Int,
    a: Int,
    modulus: Int,
    max_peaks: Int = 6,
) -> Optional[Int]:
    var probs = exponent_marginal_probs(state, n_exp, n_value)
    if len(probs) == 0:
        return None
    var indices = List[Int](capacity=len(probs))
    for i in range(len(probs)):
        indices.append(i)
    var peaks = min(max_peaks, len(indices))
    for i in range(peaks):
        var best = i
        for j in range(i + 1, len(indices)):
            if probs[indices[j]] > probs[indices[best]]:
                best = j
        var tmp = indices[i]
        indices[i] = indices[best]
        indices[best] = tmp
    for i in range(peaks):
        var candidate = estimate_order_from_measurement(
            indices[i],
            n_exp,
            a,
            modulus,
        )
        if candidate:
            return candidate
    return None


fn factors_from_order(
    a: Int,
    modulus: Int,
    r: Int,
) -> Optional[Tuple[Int, Int]]:
    if r <= 0 or (r & 1) == 1:
        return None
    var half = r // 2
    var apow = modpow(a, half, modulus)
    if apow == 1 or apow == modulus - 1:
        return None
    var f1 = gcd(apow - 1, modulus)
    var f2 = gcd(apow + 1, modulus)
    if f1 <= 1 or f2 <= 1 or f1 == modulus or f2 == modulus:
        return None
    return (f1, f2)


fn shor_factor_simulated(
    modulus: Int,
    a: Int,
    n_exp: Int,
    n_value: Int,
    max_peaks: Int = 6,
) raises -> Optional[Tuple[Int, Int]]:
    if modulus <= 1:
        return None
    var g = gcd(a, modulus)
    if g > 1:
        return (g, modulus // g)
    var qc = order_finding_circuit(n_exp, n_value, a, modulus)
    var state = QuantumState(n_exp + n_value)
    execute(state, qc, ExecContext())
    var r = estimate_order_from_state(
        state,
        n_exp,
        n_value,
        a,
        modulus,
        max_peaks=max_peaks,
    )
    if r:
        return factors_from_order(a, modulus, r.value())
    return None


fn append_order_finding(
    mut qc: QuantumCircuit,
    exp: QuantumRegister,
    value: QuantumRegister,
    a: Int,
    modulus: Int,
    swap: Bool = False,
) raises:
    """
    Order-finding circuit for Shor's algorithm.
    Assumes exp is the low-order register and value follows it.
    """
    if exp.start != 0 or value.start != exp.length:
        raise Error("Order finding expects contiguous exp+value registers")

    for i in range(exp.length):
        qc.h(exp[i])

    # Initialize |1> in the value register.
    qc.x(value[0])

    var targets = List[Int](capacity=4)
    targets.append(exp.length)
    targets.append(value.length)
    targets.append(a)
    targets.append(modulus)
    qc.add_classical("MOD_EXP", targets, apply_modexp)

    var iqft = QuantumCircuit(exp.length)
    _ = iqft_circuit(iqft, [exp.length - 1 - j for j in range(exp.length)], swap=swap)
    _ = qc.append_circuit(iqft, exp)


fn order_finding_circuit(
    n_exp: Int,
    n_value: Int,
    a: Int,
    modulus: Int,
    swap: Bool = False,
) raises -> QuantumCircuit:
    """
    Build a Shor order-finding circuit using a modular exponentiation transform.
    """
    var qc = QuantumCircuit(n_exp + n_value)
    var exp = QuantumRegister("exp", n_exp)
    var value = QuantumRegister("value", n_value, n_exp)
    append_order_finding(qc, exp, value, a, modulus, swap)
    return qc^
