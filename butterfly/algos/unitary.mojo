from butterfly.core.state import QuantumState, bit_reverse_state
from butterfly.core.types import FloatType, Type, Amplitude, Gate
from butterfly.algos.unitary_kernels import (
    Matrix4x4,
    compute_kron_product,
    unitary_radix4_kernel,
)
from memory import UnsafePointer


fn unitary_transform(mut state: QuantumState, gate: Gate):
    var n = state.size()
    var M = compute_kron_product(gate)
    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()

    recursive_unitary_entry(ptr_re, ptr_im, M, gate, n)
    bit_reverse_state(state)


fn recursive_unitary_entry(
    ptr_re: UnsafePointer[FloatType],
    ptr_im: UnsafePointer[FloatType],
    M: Matrix4x4,
    gate: Gate,
    size: Int,
):
    recursive_unitary_impl[262144](ptr_re, ptr_im, M, gate, 0, size)


fn recursive_unitary_impl[
    threshold: Int
](
    ptr_re: UnsafePointer[FloatType],
    ptr_im: UnsafePointer[FloatType],
    M: Matrix4x4,
    gate: Gate,
    start: Int,
    size: Int,
):
    if size <= threshold:
        iterative_unitary_radix4(ptr_re, ptr_im, M, gate, start, size)
        return

    var stride = size >> 2
    unitary_radix4_kernel(ptr_re, ptr_im, M, start, stride)

    recursive_unitary_impl[threshold](ptr_re, ptr_im, M, gate, start, stride)
    recursive_unitary_impl[threshold](
        ptr_re, ptr_im, M, gate, start + stride, stride
    )
    recursive_unitary_impl[threshold](
        ptr_re, ptr_im, M, gate, start + 2 * stride, stride
    )
    recursive_unitary_impl[threshold](
        ptr_re, ptr_im, M, gate, start + 3 * stride, stride
    )


fn iterative_unitary_radix4(
    ptr_re: UnsafePointer[FloatType],
    ptr_im: UnsafePointer[FloatType],
    M: Matrix4x4,
    gate: Gate,
    start: Int,
    size: Int,
):
    var m = size
    while m >= 4:
        var stride = m >> 2
        for k in range(0, size, m):
            unitary_radix4_kernel(ptr_re, ptr_im, M, start + k, stride)
        m = m >> 2

    if m == 2:
        # Single qubit case (Radix 2)
        var p_re = UnsafePointer[FloatType](ptr_re.address)
        var p_im = UnsafePointer[FloatType](ptr_im.address)

        for k in range(0, size, 2):
            var idx0 = start + k
            var idx1 = start + k + 1

            var r0 = p_re[idx0]
            var i0 = p_im[idx0]
            var r1 = p_re[idx1]
            var i1 = p_im[idx1]

            var z0_re = (
                gate[0][0].re * r0
                - gate[0][0].im * i0
                + gate[0][1].re * r1
                - gate[0][1].im * i1
            )
            var z0_im = (
                gate[0][0].re * i0
                + gate[0][0].im * r0
                + gate[0][1].re * i1
                + gate[0][1].im * r1
            )

            var z1_re = (
                gate[1][0].re * r0
                - gate[1][0].im * i0
                + gate[1][1].re * r1
                - gate[1][1].im * i1
            )
            var z1_im = (
                gate[1][0].re * i0
                + gate[1][0].im * r0
                + gate[1][1].re * i1
                + gate[1][1].im * r1
            )

            p_re[idx0] = z0_re
            p_im[idx0] = z0_im
            p_re[idx1] = z1_re
            p_im[idx1] = z1_im
