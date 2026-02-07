from collections import List
from math import sqrt, asin, cos, sin

from butterfly.core.state import QuantumState
from butterfly.core.types import Amplitude, FloatType


fn inner(v1: QuantumState, v2: QuantumState) raises -> Amplitude:
    if v1.size() != v2.size():
        raise Error("inner length mismatch")
    var sum_re = FloatType(0)
    var sum_im = FloatType(0)
    for i in range(v1.size()):
        var a = v1[i]
        var b = v2[i]
        sum_re += a.re * b.re + a.im * b.im
        sum_im += a.im * b.re - a.re * b.im
    return Amplitude(sum_re, sum_im)


fn inner(v1: List[Amplitude], v2: List[Amplitude]) raises -> Amplitude:
    if len(v1) != len(v2):
        raise Error("inner length mismatch")
    var sum_re = FloatType(0)
    var sum_im = FloatType(0)
    for i in range(len(v1)):
        var a = v1[i]
        var b = v2[i]
        sum_re += a.re * b.re + a.im * b.im
        sum_im += a.im * b.re - a.re * b.im
    return Amplitude(sum_re, sum_im)


@always_inline
fn is_close(a: FloatType, b: FloatType, tol: FloatType = FloatType(1e-6)) -> Bool:
    var diff = a - b
    if diff < 0:
        diff = -diff
    return diff <= tol


@always_inline
fn is_close_complex(
    a: Amplitude,
    b: Amplitude,
    tol: FloatType = FloatType(1e-6),
) -> Bool:
    var dr = a.re - b.re
    var di = a.im - b.im
    if dr < 0:
        dr = -dr
    if di < 0:
        di = -di
    return dr <= tol and di <= tol


fn inversion(original: QuantumState, mut current: QuantumState) raises:
    if original.size() != current.size():
        raise Error("inversion length mismatch")
    var proj = inner(original, current)
    var two = FloatType(2)
    for k in range(current.size()):
        var o_re = original.re[k]
        var o_im = original.im[k]
        var c_re = current.re[k]
        var c_im = current.im[k]
        var prod_re = proj.re * o_re - proj.im * o_im
        var prod_im = proj.re * o_im + proj.im * o_re
        current.re[k] = two * prod_re - c_re
        current.im[k] = two * prod_im - c_im


fn inversion(original: List[Amplitude], mut current: List[Amplitude]) raises:
    if len(original) != len(current):
        raise Error("inversion length mismatch")
    var proj = inner(original, current)
    var two = FloatType(2)
    for k in range(len(current)):
        var o = original[k]
        var c = current[k]
        var prod_re = proj.re * o.re - proj.im * o.im
        var prod_im = proj.re * o.im + proj.im * o.re
        current[k] = Amplitude(
            two * prod_re - c.re,
            two * prod_im - c.im,
        )


fn oracle(
    mut state: QuantumState,
    predicate: fn(Int) -> Bool,
):
    for i in range(state.size()):
        if predicate(i):
            state.re[i] = -state.re[i]
            state.im[i] = -state.im[i]


fn oracle(
    mut state: List[Amplitude],
    predicate: fn(Int) -> Bool,
):
    for i in range(len(state)):
        if predicate(i):
            var a = state[i]
            state[i] = Amplitude(-a.re, -a.im)


fn probability_on_predicate(
    state: QuantumState,
    predicate: fn(Int) -> Bool,
) -> FloatType:
    var p = FloatType(0)
    for i in range(state.size()):
        if predicate(i):
            var re = state.re[i]
            var im = state.im[i]
            p += re * re + im * im
    return p


fn probability_on_predicate(
    state: List[Amplitude],
    predicate: fn(Int) -> Bool,
) -> FloatType:
    var p = FloatType(0)
    for i in range(len(state)):
        if predicate(i):
            var a = state[i]
            p += a.re * a.re + a.im * a.im
    return p


fn grover(
    mut state: QuantumState,
    oracle: fn(mut QuantumState) raises,
    iterations: Int,
    predicate: Optional[fn(Int) -> Bool] = None,
    check: Bool = True,
) raises:
    var s = state.copy()

    var do_check = check
    if not predicate:
        do_check = False
    var theta = FloatType(0)
    if do_check:
        var p0 = probability_on_predicate(state, predicate.value())
        theta = asin(sqrt(p0))
        var proj0 = inner(s, state)
        if not is_close_complex(proj0, Amplitude(1, 0)):
            raise Error("Grover: initial state mismatch")

    for it in range(1, iterations + 1):
        oracle(state)
        inversion(s, state)

        if do_check:
            var expected = cos(FloatType(2 * it) * theta)
            var proj = inner(s, state)
            if not is_close_complex(proj, Amplitude(expected, 0)):
                raise Error("Grover: projection mismatch")

            var p_it = probability_on_predicate(state, predicate.value())
            var expected_p = sin(FloatType(2 * it + 1) * theta)
            expected_p = expected_p * expected_p
            if not is_close(p_it, expected_p):
                raise Error("Grover: probability mismatch")


fn grover(
    mut state: List[Amplitude],
    oracle: fn(mut List[Amplitude]) raises,
    iterations: Int,
    predicate: Optional[fn(Int) -> Bool] = None,
    check: Bool = True,
) raises:
    var s = state.copy()

    var do_check = check
    if not predicate:
        do_check = False
    var theta = FloatType(0)
    if do_check:
        var p0 = probability_on_predicate(state, predicate.value())
        theta = asin(sqrt(p0))
        var proj0 = inner(s, state)
        if not is_close_complex(proj0, Amplitude(1, 0)):
            raise Error("Grover: initial state mismatch")

    for it in range(1, iterations + 1):
        oracle(state)
        inversion(s, state)

        if do_check:
            var expected = cos(FloatType(2 * it) * theta)
            var proj = inner(s, state)
            if not is_close_complex(proj, Amplitude(expected, 0)):
                raise Error("Grover: projection mismatch")

            var p_it = probability_on_predicate(state, predicate.value())
            var expected_p = sin(FloatType(2 * it + 1) * theta)
            expected_p = expected_p * expected_p
            if not is_close(p_it, expected_p):
                raise Error("Grover: probability mismatch")


fn _num_qubits_from_size(size: Int) raises -> Int:
    if size <= 0 or (size & (size - 1)) != 0:
        raise Error("State size must be a power of two")
    var n = 0
    var tmp = size
    while tmp > 1:
        tmp >>= 1
        n += 1
    return n


fn _sort_targets(mut targets: List[Int]):
    var n = len(targets)
    for i in range(n):
        for j in range(i + 1, n):
            if targets[i] > targets[j]:
                var tmp = targets[i]
                targets[i] = targets[j]
                targets[j] = tmp


fn _normalize_targets(
    state: QuantumState,
    targets: List[Int],
) raises -> List[Int]:
    var n = _num_qubits_from_size(state.size())
    if len(targets) == 0:
        var full = List[Int](capacity=n)
        for i in range(n):
            full.append(i)
        return full^
    var normalized = targets.copy()
    _sort_targets(normalized)
    var seen = List[Bool](length=n, fill=False)
    for t in normalized:
        if t < 0 or t >= n:
            raise Error("Target out of bounds: " + String(t))
        if seen[t]:
            raise Error("Duplicate target: " + String(t))
        seen[t] = True
    return normalized^


fn _base_index_from_bits(
    bits: Int,
    positions: List[Int],
) -> Int:
    var base = 0
    for i in range(len(positions)):
        if (bits >> i) & 1 == 1:
            base |= 1 << positions[i]
    return base


fn apply_grover_inversion_0(
    mut state: QuantumState,
    targets: List[Int],
) raises:
    var targets_norm = _normalize_targets(state, targets)
    var n = _num_qubits_from_size(state.size())
    if len(targets_norm) == 0:
        return
    var is_target = List[Bool](length=n, fill=False)
    for t in targets_norm:
        is_target[t] = True
    var other_bits = List[Int]()
    for i in range(n):
        if not is_target[i]:
            other_bits.append(i)
    var blocks = 1 << len(other_bits)
    for block in range(blocks):
        var base = _base_index_from_bits(block, other_bits)
        state.re[base] = -state.re[base]
        state.im[base] = -state.im[base]


fn apply_grover_diffuser(
    mut state: QuantumState,
    targets: List[Int],
) raises:
    var targets_norm = _normalize_targets(state, targets)
    var n = _num_qubits_from_size(state.size())
    if len(targets_norm) == 0:
        return
    var is_target = List[Bool](length=n, fill=False)
    for t in targets_norm:
        is_target[t] = True
    var other_bits = List[Int]()
    for i in range(n):
        if not is_target[i]:
            other_bits.append(i)

    var num_targets = len(targets_norm)
    var block_size = 1 << num_targets
    var inv_block = FloatType(1) / FloatType(block_size)
    var blocks = 1 << len(other_bits)

    for block in range(blocks):
        var base = _base_index_from_bits(block, other_bits)
        var sum_re = FloatType(0)
        var sum_im = FloatType(0)
        for tmask in range(block_size):
            var idx = base
            for j in range(num_targets):
                if (tmask >> j) & 1 == 1:
                    idx |= 1 << targets_norm[j]
            sum_re += state.re[idx]
            sum_im += state.im[idx]

        var add_re = FloatType(2) * inv_block * sum_re
        var add_im = FloatType(-2) * inv_block * sum_im

        for tmask in range(block_size):
            var idx = base
            for j in range(num_targets):
                if (tmask >> j) & 1 == 1:
                    idx |= 1 << targets_norm[j]
            state.re[idx] = add_re - state.re[idx]
            state.im[idx] = add_im - state.im[idx]
