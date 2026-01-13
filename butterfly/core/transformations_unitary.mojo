from butterfly.core.state import QuantumState
from butterfly.core.types import Amplitude, FloatType
from collections import List


fn _matrix_dim(m: Int) -> Int:
    return 1 << m


fn _validate_unitary(u: List[Amplitude], m: Int) raises:
    var dim = _matrix_dim(m)
    var expected = dim * dim
    if len(u) != expected:
        raise Error(
            "Unitary size mismatch. Expected "
            + String(expected)
            + ", got "
            + String(len(u))
        )


fn apply_unitary(
    mut state: QuantumState,
    u: List[Amplitude],
    target: Int,
    m: Int,
) raises:
    """Apply a contiguous m-qubit unitary (row-major) at target."""
    _validate_unitary(u, m)
    if m <= 0:
        return

    var dim = _matrix_dim(m)
    var size = state.size()
    var n = 0
    var tmp = size
    while tmp > 1:
        tmp //= 2
        n += 1
    if target < 0 or target + m > n:
        raise Error("Target range out of bounds")

    # TODO: Make the multiplication method configurable.
    var stride = 1 << target
    var block = 1 << (target + m)
    var tmp_re = List[FloatType](length=dim, fill=0)
    var tmp_im = List[FloatType](length=dim, fill=0)
    var out_re = List[FloatType](length=dim, fill=0)
    var out_im = List[FloatType](length=dim, fill=0)

    for high in range(0, size, block):
        for low in range(stride):
            var base = high + low
            for k in range(dim):
                var idx = base + (k << target)
                tmp_re[k] = state.re[idx]
                tmp_im[k] = state.im[idx]

            for row in range(dim):
                var sum_re = FloatType(0)
                var sum_im = FloatType(0)
                var row_off = row * dim
                for col in range(dim):
                    var u_val = u[row_off + col]
                    var v_re = tmp_re[col]
                    var v_im = tmp_im[col]
                    sum_re += u_val.re * v_re - u_val.im * v_im
                    sum_im += u_val.re * v_im + u_val.im * v_re
                out_re[row] = sum_re
                out_im[row] = sum_im

            for k in range(dim):
                var idx = base + (k << target)
                state.re[idx] = out_re[k]
                state.im[idx] = out_im[k]


fn apply_controlled_unitary(
    mut state: QuantumState,
    u: List[Amplitude],
    control: Int,
    target: Int,
    m: Int,
) raises:
    """Apply a controlled contiguous m-qubit unitary at target."""
    _validate_unitary(u, m)
    if m <= 0:
        return

    var dim = _matrix_dim(m)
    var size = state.size()
    var n = 0
    var tmp = size
    while tmp > 1:
        tmp //= 2
        n += 1
    if target < 0 or target + m > n:
        raise Error("Target range out of bounds")
    if control < 0 or control >= n:
        raise Error("Control out of bounds")
    if control >= target and control < target + m:
        raise Error("Control overlaps target range")

    # TODO: Make the multiplication method configurable.
    var stride = 1 << target
    var block = 1 << (target + m)
    var tmp_re = List[FloatType](length=dim, fill=0)
    var tmp_im = List[FloatType](length=dim, fill=0)
    var out_re = List[FloatType](length=dim, fill=0)
    var out_im = List[FloatType](length=dim, fill=0)

    for high in range(0, size, block):
        for low in range(stride):
            var base = high + low
            if (base & (1 << control)) == 0:
                continue

            for k in range(dim):
                var idx = base + (k << target)
                tmp_re[k] = state.re[idx]
                tmp_im[k] = state.im[idx]

            for row in range(dim):
                var sum_re = FloatType(0)
                var sum_im = FloatType(0)
                var row_off = row * dim
                for col in range(dim):
                    var u_val = u[row_off + col]
                    var v_re = tmp_re[col]
                    var v_im = tmp_im[col]
                    sum_re += u_val.re * v_re - u_val.im * v_im
                    sum_im += u_val.re * v_im + u_val.im * v_re
                out_re[row] = sum_re
                out_im[row] = sum_im

            for k in range(dim):
                var idx = base + (k << target)
                state.re[idx] = out_re[k]
                state.im[idx] = out_im[k]
