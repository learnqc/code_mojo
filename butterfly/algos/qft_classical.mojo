from collections import List
from math import cos, sin, sqrt

from butterfly.core.state import QuantumState, apply_bit_reverse
from butterfly.core.types import FloatType, pi


fn _num_qubits_from_size(size: Int) raises -> Int:
    if size <= 0 or (size & (size - 1)) != 0:
        raise Error("State size must be a power of two")
    var n = 0
    var tmp = size
    while tmp > 1:
        tmp >>= 1
        n += 1
    return n


fn _validate_targets(
    state: QuantumState,
    targets: List[Int],
) raises -> List[Int]:
    var n = _num_qubits_from_size(state.size())
    if len(targets) == 0:
        var full = List[Int](capacity=n)
        for i in range(n):
            full.append(i)
        return full^
    var seen = List[Bool](length=n, fill=False)
    var normalized = targets.copy()
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


fn _bit_reverse_in_place(
    mut re: List[FloatType],
    mut im: List[FloatType],
):
    var n = len(re)
    if n <= 1:
        return
    var log_n = 0
    var tmp = n
    while tmp > 1:
        tmp >>= 1
        log_n += 1
    for i in range(n):
        var j = 0
        var x = i
        for _ in range(log_n):
            j = (j << 1) | (x & 1)
            x >>= 1
        if i < j:
            var tmp_re = re[i]
            var tmp_im = im[i]
            re[i] = re[j]
            im[i] = im[j]
            re[j] = tmp_re
            im[j] = tmp_im


fn _fft_in_place(
    mut re: List[FloatType],
    mut im: List[FloatType],
    inverse: Bool,
):
    var n = len(re)
    if n <= 1:
        return
    _bit_reverse_in_place(re, im)

    var sign = FloatType(1)
    if inverse:
        sign = FloatType(-1)
    var two_pi = FloatType(2) * pi

    var step = 2
    while step <= n:
        var half = step // 2
        var theta = sign * two_pi / FloatType(step)
        var w_m_re = cos(theta)
        var w_m_im = sin(theta)
        for k in range(0, n, step):
            var w_re = FloatType(1)
            var w_im = FloatType(0)
            for j in range(half):
                var t_re = w_re * re[k + j + half] - w_im * im[k + j + half]
                var t_im = w_re * im[k + j + half] + w_im * re[k + j + half]
                var u_re = re[k + j]
                var u_im = im[k + j]
                re[k + j] = u_re + t_re
                im[k + j] = u_im + t_im
                re[k + j + half] = u_re - t_re
                im[k + j + half] = u_im - t_im

                var next_w_re = w_re * w_m_re - w_im * w_m_im
                var next_w_im = w_re * w_m_im + w_im * w_m_re
                w_re = next_w_re
                w_im = next_w_im
        step *= 2


fn _apply_qft_like(
    mut state: QuantumState,
    targets: List[Int],
    inverse: Bool,
    swap: Bool,
) raises:
    var n = _num_qubits_from_size(state.size())
    var targets_norm = _validate_targets(state, targets)
    if len(targets_norm) == 0:
        return

    var lsb_first = True
    if len(targets_norm) >= 2:
        for i in range(1, len(targets_norm)):
            if targets_norm[i] <= targets_norm[i - 1]:
                lsb_first = False
                break

    var apply_pre = False
    var apply_post = False
    if inverse:
        if lsb_first:
            # Targets in LSB->MSB order: IQFT swap=False yields output bit-reversed.
            apply_post = not swap
        else:
            # Targets in MSB->LSB order: IQFT swap=False yields input bit-reversed.
            apply_pre = True
            apply_post = swap
    else:
        if lsb_first:
            # QFT is inverse of the above IQFT ordering.
            apply_pre = not swap
        else:
            apply_pre = swap
            apply_post = True

    if apply_pre:
        if len(targets_norm) == n:
            apply_bit_reverse(state, List[Int]())
        else:
            apply_bit_reverse(state, targets_norm)
    var is_target = List[Bool](length=n, fill=False)
    for t in targets_norm:
        is_target[t] = True
    var other_bits = List[Int]()
    for i in range(n):
        if not is_target[i]:
            other_bits.append(i)

    var num_targets = len(targets_norm)
    var block_size = 1 << num_targets
    var scale = FloatType(1) / sqrt(FloatType(block_size))
    var blocks = 1 << len(other_bits)

    var tmp_re = List[FloatType](length=block_size, fill=0)
    var tmp_im = List[FloatType](length=block_size, fill=0)

    for block in range(blocks):
        var base = _base_index_from_bits(block, other_bits)
        for tmask in range(block_size):
            var idx = base
            for j in range(num_targets):
                if (tmask >> j) & 1 == 1:
                    if lsb_first:
                        idx |= 1 << targets_norm[j]
                    else:
                        idx |= 1 << targets_norm[num_targets - 1 - j]
            tmp_re[tmask] = state.re[idx]
            tmp_im[tmask] = state.im[idx]

        _fft_in_place(tmp_re, tmp_im, inverse)

        for tmask in range(block_size):
            var idx = base
            for j in range(num_targets):
                if (tmask >> j) & 1 == 1:
                    if lsb_first:
                        idx |= 1 << targets_norm[j]
                    else:
                        idx |= 1 << targets_norm[num_targets - 1 - j]
            state.re[idx] = tmp_re[tmask] * scale
            state.im[idx] = tmp_im[tmask] * scale

    if apply_post:
        if len(targets_norm) == n:
            apply_bit_reverse(state, List[Int]())
        else:
            apply_bit_reverse(state, targets_norm)


fn apply_qft_classical(
    mut state: QuantumState,
    targets: List[Int],
    swap: Bool = False,
) raises:
    _apply_qft_like(state, targets, inverse=False, swap=swap)


fn apply_iqft_classical(
    mut state: QuantumState,
    targets: List[Int],
    swap: Bool = False,
) raises:
    _apply_qft_like(state, targets, inverse=True, swap=swap)
