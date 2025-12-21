from butterfly.core.state import QuantumState, bit_reverse_state
from butterfly.core.types import FloatType, Type
from memory import UnsafePointer
from sys.info import simd_width_of
from algorithm import parallelize, vectorize
from math import sqrt

# Compile-time constants
alias simd_width = simd_width_of[FloatType]()


fn hadamard_transform(mut state: QuantumState):
    """
    Applies the Walsh-Hadamard Transform (WHT) to the quantum state.
    """
    var n = state.size()
    # Initial pointers for recursive transform
    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()

    recursive_wht_entry(ptr_re, n)
    recursive_wht_entry(ptr_im, n)

    bit_reverse_state(state)

    # Re-acquire pointers because bit_reverse_state allocates new buffers
    ptr_re = state.re.unsafe_ptr()
    ptr_im = state.im.unsafe_ptr()

    var scale = FloatType(1.0) / sqrt(FloatType(n))
    for i in range(n):
        ptr_re[i] *= scale
        ptr_im[i] *= scale


fn recursive_wht_entry(ptr: UnsafePointer[FloatType], size: Int):
    recursive_wht_impl[262144](ptr, 0, size)


fn recursive_wht_impl[
    threshold: Int
](ptr: UnsafePointer[FloatType], start: Int, size: Int):
    if size <= threshold:
        iterative_wht_radix4(ptr, start, size)
        return

    var stride = size >> 2
    hadamard_radix4_kernel(ptr, start, stride)

    recursive_wht_impl[threshold](ptr, start, stride)
    recursive_wht_impl[threshold](ptr, start + stride, stride)
    recursive_wht_impl[threshold](ptr, start + 2 * stride, stride)
    recursive_wht_impl[threshold](ptr, start + 3 * stride, stride)


fn hadamard_radix4_kernel(
    ptr: UnsafePointer[FloatType], start: Int, stride: Int
):
    # Launder the pointer to ensure mutability and bypass read-only constraints on arguments
    var p = UnsafePointer[FloatType](ptr.address)

    @parameter
    fn v_kernel[w: Int](idx: Int):
        var k = idx
        # Calculate shifted addresses relative to laundered pointer
        var addr0 = p + (start + k)
        var addr1 = p + (start + stride + k)
        var addr2 = p + (start + 2 * stride + k)
        var addr3 = p + (start + 3 * stride + k)

        # Bitcast to SIMD pointers
        var p0_v = addr0.bitcast[SIMD[DType.float64, w]]()
        var p1_v = addr1.bitcast[SIMD[DType.float64, w]]()
        var p2_v = addr2.bitcast[SIMD[DType.float64, w]]()
        var p3_v = addr3.bitcast[SIMD[DType.float64, w]]()

        # Load
        var v0 = p0_v[0]
        var v1 = p1_v[0]
        var v2 = p2_v[0]
        var v3 = p3_v[0]

        var t1 = v0 + v2
        var t2 = v0 - v2
        var t3 = v1 + v3
        var t4 = v1 - v3

        var z0 = t1 + t3
        var z1 = t1 - t3
        var z2 = t2 + t4
        var z3 = t2 - t4

        # Store
        p0_v[0] = z0
        p1_v[0] = z1
        p2_v[0] = z2
        p3_v[0] = z3

    vectorize[v_kernel, simd_width](stride)


fn iterative_wht_radix4(ptr: UnsafePointer[FloatType], start: Int, size: Int):
    # Launder pointer
    var p = UnsafePointer[FloatType](ptr.address)
    var m = size
    while m >= 4:
        var stride = m >> 2
        for k in range(0, size, m):
            hadamard_radix4_kernel(ptr, start + k, stride)
        m = m >> 2

    if m == 2:
        for k in range(0, size, 2):
            var idx0 = start + k
            var idx1 = idx0 + 1

            var v0 = p[idx0]
            var v1 = p[idx1]

            p[idx0] = v0 + v1
            p[idx1] = v0 - v1
