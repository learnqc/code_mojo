from butterfly.core.state import QuantumState
from butterfly.core.types import FloatType, Gate, Amplitude
from sys.info import simd_width_of

alias simd_width = simd_width_of[Type]()
from butterfly.algos.unitary_kernels import (
    Matrix4x4,
    compute_kron_product,
    unitary_radix4_kernel,
)
from memory import UnsafePointer
from algorithm import parallelize, vectorize
from math import min


from butterfly.algos.vec_swaps import swap_bits_simd


fn unitary_natural_transform(mut state: QuantumState, gate: Gate):
    var size = state.size()
    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()
    unitary_natural_transform_ptr(ptr_re, ptr_im, size, gate)


fn controlled_unitary_natural_transform(
    mut state: QuantumState, gate: Gate, control: Int
):
    var n = Int(log2(Float64(state.size())))
    var last_qubit = n - 1

    # 1. Swap control to MSB (if needed)
    if control != last_qubit:
        swap_bits_simd(state, control, last_qubit)

    # 2. Apply Unitary loop on the SECOND HALF of the state (where MSB=1)
    # The effective size is N/2. The effective qubits are n-1.
    # The 'gate' (assumed 2x2) is applied to all n-1 qubits.
    var half_size = state.size() >> 1
    var ptr_re = state.re.unsafe_ptr().offset(half_size)
    var ptr_im = state.im.unsafe_ptr().offset(half_size)

    unitary_natural_transform_ptr(ptr_re, ptr_im, half_size, gate)

    # 3. Swap back
    if control != last_qubit:
        swap_bits_simd(state, control, last_qubit)


fn unitary_natural_transform_ptr(
    ptr_re: UnsafePointer[FloatType],
    ptr_im: UnsafePointer[FloatType],
    size: Int,
    gate: Gate,
):
    from math import log2

    var n = Int(log2(Float64(size)))
    var M = compute_kron_product(gate)

    alias BLOCK_LOG = 11
    var m = n
    if BLOCK_LOG < n:
        m = BLOCK_LOG

    # 1. Fused Local Phase
    var num_blocks = size >> m

    @parameter
    fn block_worker(block_idx: Int):
        var start = block_idx << m
        fused_block_transform(ptr_re, ptr_im, M, gate, start, m)

    parallelize[block_worker](num_blocks)

    # 2. Global Phase
    if n > m:
        for t in range(m, n, 2):
            if t + 1 < n:
                var stride = 1 << t
                apply_global_radix4(ptr_re, ptr_im, M, size, stride)
            else:
                var stride = 1 << t
                apply_global_radix2(ptr_re, ptr_im, gate, size, stride)


fn fused_block_transform(
    ptr_re: UnsafePointer[FloatType],
    ptr_im: UnsafePointer[FloatType],
    M: Matrix4x4,
    gate: Gate,
    start: Int,
    num_dims: Int,
):
    var blk_size = 1 << num_dims
    var t = 0
    while t < num_dims:
        if t + 1 < num_dims:
            var stride = 1 << t
            var step = stride << 2
            for k in range(0, blk_size, step):
                unitary_radix4_kernel(ptr_re, ptr_im, M, start + k, stride)
            t += 2
        else:
            var stride = 1 << t
            var step = stride << 1
            for k in range(0, blk_size, step):
                unitary_radix2_kernel_local(
                    ptr_re, ptr_im, gate, start + k, stride
                )
            t += 1


fn apply_global_radix4(
    ptr_re: UnsafePointer[FloatType],
    ptr_im: UnsafePointer[FloatType],
    M: Matrix4x4,
    size: Int,
    stride: Int,
):
    var step = stride << 2

    @parameter
    fn global_worker(idx: Int):
        var k = idx * step
        unitary_radix4_kernel(ptr_re, ptr_im, M, k, stride)

    parallelize[global_worker](size // step)


fn apply_global_radix2(
    ptr_re: UnsafePointer[FloatType],
    ptr_im: UnsafePointer[FloatType],
    gate: Gate,
    size: Int,
    stride: Int,
):
    var step = stride << 1

    @parameter
    fn global_worker_radix2(idx: Int):
        var k = idx * step
        unitary_radix2_kernel_local(ptr_re, ptr_im, gate, k, stride)

    parallelize[global_worker_radix2](size // step)


fn unitary_radix2_kernel_local(
    ptr_re: UnsafePointer[FloatType],
    ptr_im: UnsafePointer[FloatType],
    gate: Gate,
    start: Int,
    stride: Int,
):
    var p_re = UnsafePointer[FloatType](ptr_re.address)
    var p_im = UnsafePointer[FloatType](ptr_im.address)

    var g00 = gate[0][0]
    var g01 = gate[0][1]
    var g10 = gate[1][0]
    var g11 = gate[1][1]

    @parameter
    fn v_kernel[w: Int](idx: Int):
        var idx0 = start + idx
        var idx1 = start + stride + idx

        var r0 = p_re.load[width=w](idx0)
        var i0 = p_im.load[width=w](idx0)
        var r1 = p_re.load[width=w](idx1)
        var i1 = p_im.load[width=w](idx1)

        var z0_re = SIMD[DType.float64, w](0.0)
        var z0_im = SIMD[DType.float64, w](0.0)
        acc_mul_simple(z0_re, z0_im, g00, r0, i0)
        acc_mul_simple(z0_re, z0_im, g01, r1, i1)

        var z1_re = SIMD[DType.float64, w](0.0)
        var z1_im = SIMD[DType.float64, w](0.0)
        acc_mul_simple(z1_re, z1_im, g10, r0, i0)
        acc_mul_simple(z1_re, z1_im, g11, r1, i1)

        p_re.store(idx0, z0_re)
        p_im.store(idx0, z0_im)
        p_re.store(idx1, z1_re)
        p_im.store(idx1, z1_im)

    vectorize[v_kernel, simd_width](stride)


@always_inline
fn acc_mul_simple[
    w: Int
](
    mut acc_re: SIMD[DType.float64, w],
    mut acc_im: SIMD[DType.float64, w],
    scalar: Amplitude,
    vec_re: SIMD[DType.float64, w],
    vec_im: SIMD[DType.float64, w],
):
    var a = scalar.re
    var b = scalar.im
    var c = vec_re
    var d = vec_im
    acc_re += a * c - b * d
    acc_im += a * d + b * c
