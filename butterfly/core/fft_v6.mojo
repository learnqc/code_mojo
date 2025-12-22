"""
FFT V5: Subspace-based Partial Classical Fourier Transform.
Optimized for contiguous qubit targets (registers).
Treats a k-qubit transform as 2^{N-k} independent 2^k-point FFTs.
"""
from math import pi, cos, sin, sqrt, log2
from memory import UnsafePointer
from algorithm import parallelize, vectorize
from sys.info import simd_width_of
from butterfly.core.types import FloatType, Type
from butterfly.core.state import QuantumState, bit_reverse_state
from butterfly.core.fft_v4 import fft_v4


fn cft(
    mut state: QuantumState,
    targets: List[Int],
    inverse: Bool = False,
    do_swap: Bool = True,
):
    """
    Apply Classical Fourier Transform (CFT) to the target qubits.
    Delegates to v4 for full-range forward transforms, and v5 for partial/inverse.
    """
    var n = state.size()

    if 1 << len(targets) == n:
        fft_v4(state)
    else:
        cft_v6_contiguous(state, targets, inverse, do_swap)


alias simd_width = simd_width_of[FloatType]()


@always_inline
fn compute_twiddle[
    inverse: Bool
](j: Int, n: Int) -> Tuple[FloatType, FloatType]:
    """Compute on-the-fly twiddle factor W_n^j."""
    alias angle_base = -2.0 * pi if not inverse else 2.0 * pi
    var angle = angle_base * Float64(j) / Float64(n)
    return cos(angle), sin(angle)


fn cft_v6_contiguous(
    mut state: QuantumState,
    targets: List[Int],
    inverse: Bool = False,
    do_swap: Bool = True,
):
    """
    Apply CFT to a contiguous set of qubits using the v5 subspace approach.
    Assumes targets are sorted and contiguous: [b, b+1, ..., b+k-1].
    """
    var k = len(targets)
    if k == 0:
        return

    var n_total = state.size()
    var b = targets[0]
    var stride = 1 << b
    var span = 1 << k
    var num_subspaces = n_total // span

    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()
    alias scale = sqrt(0.5).cast[Type]()

    # Precompute Twiddle Table for k >= 4
    # Size: (span / 2) floats each for Re and Im
    var twiddle_storage_re = List[FloatType](capacity=1)
    var twiddle_storage_im = List[FloatType](capacity=1)
    twiddle_storage_re.append(0.0)
    twiddle_storage_im.append(0.0)
    var twiddle_ptr_re = twiddle_storage_re.unsafe_ptr()
    var twiddle_ptr_im = twiddle_storage_im.unsafe_ptr()

    if k >= 4:
        var half_span = span >> 1
        twiddle_storage_re = List[FloatType](capacity=half_span)
        twiddle_storage_im = List[FloatType](capacity=half_span)
        for _ in range(half_span):
            twiddle_storage_re.append(0.0)
            twiddle_storage_im.append(0.0)

        twiddle_ptr_re = twiddle_storage_re.unsafe_ptr()
        twiddle_ptr_im = twiddle_storage_im.unsafe_ptr()

        alias two_pi = 2.0 * pi
        var angle_base = two_pi if not inverse else -two_pi
        var base_n = Float64(span)

        for j in range(half_span):
            var angle = angle_base * Float64(j) / base_n
            twiddle_ptr_re[j] = cos(angle)
            twiddle_ptr_im[j] = sin(angle)

    @parameter
    fn subspace_worker(blk_idx: Int):
        # Calculate base offset using bit-insertion
        var low_mask = stride - 1
        var offset = (blk_idx & low_mask) | ((blk_idx & ~low_mask) << k)

        if k == 1:
            var i0 = offset
            var i1 = offset + stride
            var u_re = ptr_re[i0]
            var u_im = ptr_im[i0]
            var v_re = ptr_re[i1]
            var v_im = ptr_im[i1]
            ptr_re[i0] = (u_re + v_re) * scale
            ptr_im[i0] = (u_im + v_im) * scale
            ptr_re[i1] = (u_re - v_re) * scale
            ptr_im[i1] = (u_im - v_im) * scale
            return

        if k == 2:
            var i0 = offset
            var i1 = offset + stride
            var i2 = offset + 2 * stride
            var i3 = offset + 3 * stride
            var a0_re = ptr_re[i0]
            var a0_im = ptr_im[i0]
            var a1_re = ptr_re[i1]
            var a1_im = ptr_im[i1]
            var a2_re = ptr_re[i2]
            var a2_im = ptr_im[i2]
            var a3_re = ptr_re[i3]
            var a3_im = ptr_im[i3]

            if not inverse:
                if do_swap:
                    var t_re = a1_re
                    a1_re = a2_re
                    a2_re = t_re
                    var t_im = a1_im
                    a1_im = a2_im
                    a2_im = t_im
                var r0_re = (a0_re + a1_re) * scale
                var r0_im = (a0_im + a1_im) * scale
                var d0_re = (a0_re - a1_re) * scale
                var d0_im = (a0_im - a1_im) * scale
                var r1_re = (a2_re + a3_re) * scale
                var r1_im = (a2_im + a3_im) * scale
                var d1_re = (a2_re - a3_re) * scale
                var d1_im = (a2_im - a3_im) * scale
                ptr_re[i0] = (r0_re + r1_re) * scale
                ptr_im[i0] = (r0_im + r1_im) * scale
                ptr_re[i2] = (r0_re - r1_re) * scale
                ptr_im[i2] = (r0_im - r1_im) * scale
                var v1_rot_re = -d1_im
                var v1_rot_im = d1_re
                ptr_re[i1] = (d0_re + v1_rot_re) * scale
                ptr_im[i1] = (d0_im + v1_rot_im) * scale
                ptr_re[i3] = (d0_re - v1_rot_re) * scale
                ptr_im[i3] = (d0_im - v1_rot_im) * scale
            else:
                var r0_re = (a0_re + a2_re) * scale
                var r0_im = (a0_im + a2_im) * scale
                var d0_re = (a0_re - a2_re) * scale
                var d0_im = (a0_im - a2_im) * scale
                var r1_re = (a1_re + a3_re) * scale
                var r1_im = (a1_im + a3_im) * scale
                var d1_re = (a1_re - a3_re) * scale
                var d1_im = (a1_im - a3_im) * scale
                var v1_rot_re = d1_im
                var v1_rot_im = -d1_re  # W4^1_inv = -i
                var res0_re = (r0_re + r1_re) * scale
                var res0_im = (r0_im + r1_im) * scale
                var res1_re = (r0_re - r1_re) * scale
                var res1_im = (r0_im - r1_im) * scale
                var res2_re = (d0_re + v1_rot_re) * scale
                var res2_im = (d0_im + v1_rot_im) * scale
                var res3_re = (d0_re - v1_rot_re) * scale
                var res3_im = (d0_im - v1_rot_im) * scale
                if do_swap:
                    ptr_re[i0] = res0_re
                    ptr_im[i0] = res0_im
                    ptr_re[i1] = res2_re
                    ptr_im[i1] = res2_im
                    ptr_re[i2] = res1_re
                    ptr_im[i2] = res1_im
                    ptr_re[i3] = res3_re
                    ptr_im[i3] = res3_im
                else:
                    ptr_re[i0] = res0_re
                    ptr_im[i0] = res0_im
                    ptr_re[i1] = res1_re
                    ptr_im[i1] = res1_im
                    ptr_re[i2] = res2_re
                    ptr_im[i2] = res2_im
                    ptr_re[i3] = res3_re
                    ptr_im[i3] = res3_im
            return

        if k == 3:
            var i0 = offset
            var i1 = offset + stride
            var i2 = offset + 2 * stride
            var i3 = offset + 3 * stride
            var i4 = offset + 4 * stride
            var i5 = offset + 5 * stride
            var i6 = offset + 6 * stride
            var i7 = offset + 7 * stride
            var a0_re = ptr_re[i0]
            var a0_im = ptr_im[i0]
            var a1_re = ptr_re[i1]
            var a1_im = ptr_im[i1]
            var a2_re = ptr_re[i2]
            var a2_im = ptr_im[i2]
            var a3_re = ptr_re[i3]
            var a3_im = ptr_im[i3]
            var a4_re = ptr_re[i4]
            var a4_im = ptr_im[i4]
            var a5_re = ptr_re[i5]
            var a5_im = ptr_im[i5]
            var a6_re = ptr_re[i6]
            var a6_im = ptr_im[i6]
            var a7_re = ptr_re[i7]
            var a7_im = ptr_im[i7]

            if not inverse:
                if do_swap:
                    var t_re = a1_re
                    a1_re = a4_re
                    a4_re = t_re
                    var t_im = a1_im
                    a1_im = a4_im
                    a4_im = t_im
                    t_re = a3_re
                    a3_re = a6_re
                    a6_re = t_re
                    t_im = a3_im
                    a3_im = a6_im
                    a6_im = t_im
                var s0r0_re = (a0_re + a1_re) * scale
                var s0r0_im = (a0_im + a1_im) * scale
                var s0d0_re = (a0_re - a1_re) * scale
                var s0d0_im = (a0_im - a1_im) * scale
                var s0r1_re = (a2_re + a3_re) * scale
                var s0r1_im = (a2_im + a3_im) * scale
                var s0d1_re = (a2_re - a3_re) * scale
                var s0d1_im = (a2_im - a3_im) * scale
                var s0r2_re = (a4_re + a5_re) * scale
                var s0r2_im = (a4_im + a5_im) * scale
                var s0d2_re = (a4_re - a5_re) * scale
                var s0d2_im = (a4_im - a5_im) * scale
                var s0r3_re = (a6_re + a7_re) * scale
                var s0r3_im = (a6_im + a7_im) * scale
                var s0d3_re = (a6_re - a7_re) * scale
                var s0d3_im = (a6_im - a7_im) * scale
                var s1r0_re = (s0r0_re + s0r1_re) * scale
                var s1r0_im = (s0r0_im + s0r1_im) * scale
                var s1d0_re = (s0r0_re - s0r1_re) * scale
                var s1d0_im = (s0r0_im - s0r1_im) * scale
                var v1_rot_re = -s0d1_im
                var v1_rot_im = s0d1_re
                var s1r1_re = (s0d0_re + v1_rot_re) * scale
                var s1r1_im = (s0d0_im + v1_rot_im) * scale
                var s1d1_re = (s0d0_re - v1_rot_re) * scale
                var s1d1_im = (s0d0_im - v1_rot_im) * scale
                var s1r2_re = (s0r2_re + s0r3_re) * scale
                var s1r2_im = (s0r2_im + s0r3_im) * scale
                var s1d2_re = (s0r2_re - s0r3_re) * scale
                var s1d2_im = (s0r2_im - s0r3_im) * scale
                var v3_rot_re = -s0d3_im
                var v3_rot_im = s0d3_re
                var s1r3_re = (s0d2_re + v3_rot_re) * scale
                var s1r3_im = (s0d2_im + v3_rot_im) * scale
                var s1d3_re = (s0d2_re - v3_rot_re) * scale
                var s1d3_im = (s0d2_im - v3_rot_im) * scale
                ptr_re[i0] = (s1r0_re + s1r2_re) * scale
                ptr_im[i0] = (s1r0_im + s1r2_im) * scale
                ptr_re[i4] = (s1r0_re - s1r2_re) * scale
                ptr_im[i4] = (s1r0_im - s1r2_im) * scale
                var v5_rot_re = (s1r3_re - s1r3_im) * scale
                var v5_rot_im = (s1r3_re + s1r3_im) * scale
                ptr_re[i1] = (s1r1_re + v5_rot_re) * scale
                ptr_im[i1] = (s1r1_im + v5_rot_im) * scale
                ptr_re[i5] = (s1r1_re - v5_rot_re) * scale
                ptr_im[i5] = (s1r1_im - v5_rot_im) * scale
                var v6_rot_re = -s1d2_im
                var v6_rot_im = s1d2_re
                ptr_re[i2] = (s1d0_re + v6_rot_re) * scale
                ptr_im[i2] = (s1d0_im + v6_rot_im) * scale
                ptr_re[i6] = (s1d0_re - v6_rot_re) * scale
                ptr_im[i6] = (s1d0_im - v6_rot_im) * scale
                var v7_rot_re = (-s1d3_re - s1d3_im) * scale
                var v7_rot_im = (s1d3_re - s1d3_im) * scale
                ptr_re[i3] = (s1d1_re + v7_rot_re) * scale
                ptr_im[i3] = (s1d1_im + v7_rot_im) * scale
                ptr_re[i7] = (s1d1_re - v7_rot_re) * scale
                ptr_im[i7] = (s1d1_im - v7_rot_im) * scale
            else:
                var s2r0_re = (a0_re + a4_re) * scale
                var s2r0_im = (a0_im + a4_im) * scale
                var s2d0_re = (a0_re - a4_re) * scale
                var s2d0_im = (a0_im - a4_im) * scale
                var s2r1_re = (a1_re + a5_re) * scale
                var s2r1_im = (a1_im + a5_im) * scale
                var s2d1_re = (a1_re - a5_re) * scale
                var s2d1_im = (a1_im - a5_im) * scale
                var s2r2_re = (a2_re + a6_re) * scale
                var s2r2_im = (a2_im + a6_im) * scale
                var s2d2_re = (a2_re - a6_re) * scale
                var s2d2_im = (a2_im - a6_im) * scale
                var s2r3_re = (a3_re + a7_re) * scale
                var s2r3_im = (a3_im + a7_im) * scale
                var s2d3_re = (a3_re - a7_re) * scale
                var s2d3_im = (a3_im - a7_im) * scale
                var s2d1_rot_re = (s2d1_re + s2d1_im) * scale
                var s2d1_rot_im = (s2d1_im - s2d1_re) * scale
                var s2d2_rot_re = s2d2_im
                var s2d2_rot_im = -s2d2_re
                var s2d3_rot_re = (s2d3_im - s2d3_re) * scale
                var s2d3_rot_im = (-s2d3_re - s2d3_im) * scale
                var s1r0_re = (s2r0_re + s2r2_re) * scale
                var s1r0_im = (s2r0_im + s2r2_im) * scale
                var s1d0_re = (s2r0_re - s2r2_re) * scale
                var s1d0_im = (s2r0_im - s2r2_im) * scale
                var s1r1_re = (s2r1_re + s2r3_re) * scale
                var s1r1_im = (s2r1_im + s2r3_im) * scale
                var s1d1_re = (s2r1_re - s2r3_re) * scale
                var s1d1_im = (s2r1_im - s2r3_im) * scale
                var s1r2_re = (s2d0_re + s2d2_rot_re) * scale
                var s1r2_im = (s2d0_im + s2d2_rot_im) * scale
                var s1d2_re = (s2d0_re - s2d2_rot_re) * scale
                var s1d2_im = (s2d0_im - s2d2_rot_im) * scale
                var s1r3_re = (s2d1_rot_re + s2d3_rot_re) * scale
                var s1r3_im = (s2d1_rot_im + s2d3_rot_im) * scale
                var s1d3_re = (s2d1_rot_re - s2d3_rot_re) * scale
                var s1d3_im = (s2d1_rot_im - s2d3_rot_im) * scale
                var s1d1_rot_re = s1d1_im
                var s1d1_rot_im = -s1d1_re
                var s1d3_rot_re = s1d3_im
                var s1d3_rot_im = -s1d3_re
                var res0_re = (s1r0_re + s1r1_re) * scale
                var res0_im = (s1r0_im + s1r1_im) * scale
                var res1_re = (s1r0_re - s1r1_re) * scale
                var res1_im = (s1r0_im - s1r1_im) * scale
                var res2_re = (s1d0_re + s1d1_rot_re) * scale
                var res2_im = (s1d0_im + s1d1_rot_im) * scale
                var res3_re = (s1d0_re - s1d1_rot_re) * scale
                var res3_im = (s1d0_im - s1d1_rot_im) * scale
                var res4_re = (s1r2_re + s1r3_re) * scale
                var res4_im = (s1r2_im + s1r3_im) * scale
                var res5_re = (s1r2_re - s1r3_re) * scale
                var res5_im = (s1r2_im - s1r3_im) * scale
                var res6_re = (s1d2_re + s1d3_rot_re) * scale
                var res6_im = (s1d2_im + s1d3_rot_im) * scale
                var res7_re = (s1d2_re - s1d3_rot_re) * scale
                var res7_im = (s1d2_im - s1d3_rot_im) * scale
                if do_swap:
                    ptr_re[i0] = res0_re
                    ptr_im[i0] = res0_im
                    ptr_re[i1] = res4_re
                    ptr_im[i1] = res4_im
                    ptr_re[i2] = res2_re
                    ptr_im[i2] = res2_im
                    ptr_re[i3] = res6_re
                    ptr_im[i3] = res6_im
                    ptr_re[i4] = res1_re
                    ptr_im[i4] = res1_im
                    ptr_re[i5] = res5_re
                    ptr_im[i5] = res5_im
                    ptr_re[i6] = res3_re
                    ptr_im[i6] = res3_im
                    ptr_re[i7] = res7_re
                    ptr_im[i7] = res7_im
                else:
                    ptr_re[i0] = res0_re
                    ptr_im[i0] = res0_im
                    ptr_re[i1] = res1_re
                    ptr_im[i1] = res1_im
                    ptr_re[i2] = res2_re
                    ptr_im[i2] = res2_im
                    ptr_re[i3] = res3_re
                    ptr_im[i3] = res3_im
                    ptr_re[i4] = res4_re
                    ptr_im[i4] = res4_im
                    ptr_re[i5] = res5_re
                    ptr_im[i5] = res5_im
                    ptr_re[i6] = res6_re
                    ptr_im[i6] = res6_im
                    ptr_re[i7] = res7_re
                    ptr_im[i7] = res7_im
            return

        if not inverse:
            # Forward QFT: Swap(start) then DIT(0..k-1)
            # 1. Local Swap at START
            if do_swap:
                for i in range(span):
                    var reversed_i = 0
                    for b_idx in range(k):
                        if (i >> b_idx) & 1:
                            reversed_i |= 1 << (k - 1 - b_idx)

                    if i < reversed_i:
                        var idx_i = offset + i * stride
                        var idx_r = offset + reversed_i * stride
                        var tmp_re = ptr_re[idx_i]
                        var tmp_im = ptr_im[idx_i]
                        ptr_re[idx_i] = ptr_re[idx_r]
                        ptr_im[idx_i] = ptr_im[idx_r]
                        ptr_re[idx_r] = tmp_re
                        ptr_im[idx_r] = tmp_im

            # 2. DIT Stages
            for stage in range(k):
                var s_stride = 1 << stage
                var s_step = s_stride << 1
                var n_points = s_step

                for group in range(0, span, s_step):

                    @parameter
                    fn butterfly_vectorized[simd_width: Int](j: Int):
                        var idx0 = offset + (group + j) * stride
                        var idx1 = idx0 + s_stride * stride

                        var w_re = SIMD[Type, simd_width]()
                        var w_im = SIMD[Type, simd_width]()
                        if k >= 4:
                            var shift = k - 1 - stage
                            if shift == 0:
                                w_re = twiddle_ptr_re.load[width=simd_width](j)
                                w_im = twiddle_ptr_im.load[width=simd_width](j)
                            else:
                                for v_idx in range(simd_width):
                                    var tidx = (j + v_idx) << shift
                                    w_re[v_idx] = twiddle_ptr_re[tidx]
                                    w_im[v_idx] = twiddle_ptr_im[tidx]
                        else:
                            for v_idx in range(simd_width):
                                var w = compute_twiddle[True](
                                    j + v_idx, n_points
                                )
                                w_re[v_idx] = w[0]
                                w_im[v_idx] = w[1]

                        var u_re = ptr_re.load[width=simd_width](idx0)
                        var u_im = ptr_im.load[width=simd_width](idx0)
                        var v_re = ptr_re.load[width=simd_width](idx1)
                        var v_im = ptr_im.load[width=simd_width](idx1)

                        var v_rot_re = v_re * w_re - v_im * w_im
                        var v_rot_im = v_re * w_im + v_im * w_re

                        ptr_re.store(idx0, (u_re + v_rot_re) * scale)
                        ptr_im.store(idx0, (u_im + v_rot_im) * scale)
                        ptr_re.store(idx1, (u_re - v_rot_re) * scale)
                        ptr_im.store(idx1, (u_im - v_rot_im) * scale)

                    vectorize[butterfly_vectorized, simd_width](s_stride)
        else:
            # Inverse QFT: DIF(k-1..0, neg) then Swap(end)
            # 1. DIF Stages
            for stage in reversed(range(k)):
                var s_stride = 1 << stage
                var s_step = s_stride << 1
                var n_points = s_step

                for group in range(0, span, s_step):

                    @parameter
                    fn butterfly_vectorized_inv[simd_width: Int](j: Int):
                        var idx0 = offset + (group + j) * stride
                        var idx1 = idx0 + s_stride * stride

                        var w_re = SIMD[Type, simd_width]()
                        var w_im = SIMD[Type, simd_width]()
                        if k >= 4:
                            var shift = k - 1 - stage
                            if shift == 0:
                                w_re = twiddle_ptr_re.load[width=simd_width](j)
                                w_im = twiddle_ptr_im.load[width=simd_width](j)
                            else:
                                for v_idx in range(simd_width):
                                    var tidx = (j + v_idx) << shift
                                    w_re[v_idx] = twiddle_ptr_re[tidx]
                                    w_im[v_idx] = twiddle_ptr_im[tidx]
                        else:
                            for v_idx in range(simd_width):
                                var w = compute_twiddle[False](
                                    j + v_idx, n_points
                                )
                                w_re[v_idx] = w[0]
                                w_im[v_idx] = w[1]

                        var u_re = ptr_re.load[width=simd_width](idx0)
                        var u_im = ptr_im.load[width=simd_width](idx0)
                        var v_re = ptr_re.load[width=simd_width](idx1)
                        var v_im = ptr_im.load[width=simd_width](idx1)

                        # DIF: Butterfly then Rotate v
                        var res_re = (u_re + v_re) * scale
                        var res_im = (u_im + v_im) * scale
                        var diff_re = (u_re - v_re) * scale
                        var diff_im = (u_im - v_im) * scale

                        ptr_re.store(idx0, res_re)
                        ptr_im.store(idx0, res_im)
                        ptr_re.store(idx1, diff_re * w_re - diff_im * w_im)
                        ptr_im.store(idx1, diff_re * w_im + diff_im * w_re)

                    vectorize[butterfly_vectorized_inv, simd_width](s_stride)

            # 2. Local Swap at END
            if do_swap:
                for i in range(span):
                    var reversed_i = 0
                    for b_idx in range(k):
                        if (i >> b_idx) & 1:
                            reversed_i |= 1 << (k - 1 - b_idx)

                    if i < reversed_i:
                        var idx_i = offset + i * stride
                        var idx_r = offset + reversed_i * stride
                        var tmp_re = ptr_re[idx_i]
                        var tmp_im = ptr_im[idx_i]
                        ptr_re[idx_i] = ptr_re[idx_r]
                        ptr_im[idx_i] = ptr_im[idx_r]
                        ptr_re[idx_r] = tmp_re
                        ptr_im[idx_r] = tmp_im

    if num_subspaces > 1:
        parallelize[subspace_worker](num_subspaces)
    else:
        # Single subspace case (e.g. Full range k=N)
        # We must parallelize the inner loops to use all cores.
        var offset = 0
        if not inverse:
            if do_swap:
                # Parallelize Swap
                @parameter
                fn swap_worker(i: Int):
                    var reversed_i = 0
                    for b_idx in range(k):
                        if (i >> b_idx) & 1:
                            reversed_i |= 1 << (k - 1 - b_idx)
                    if i < reversed_i:
                        var idx_i = offset + i * stride
                        var idx_r = offset + reversed_i * stride
                        var tmp_re = ptr_re[idx_i]
                        var tmp_im = ptr_im[idx_i]
                        ptr_re[idx_i] = ptr_re[idx_r]
                        ptr_im[idx_i] = ptr_im[idx_r]
                        ptr_re[idx_r] = tmp_re
                        ptr_im[idx_r] = tmp_im

                parallelize[swap_worker](span)

                for stage in range(k):
                    var s_stride = 1 << stage
                    var s_step = s_stride << 1
                    var n_points = s_step
                    var num_groups = span // s_step
                    var shift = k - 1 - stage

                    if num_groups >= 16 or s_stride < simd_width:

                        @parameter
                        fn group_worker(group_idx: Int):
                            var group = group_idx * s_step

                            @parameter
                            fn butterfly_vectorized[simd_width: Int](j: Int):
                                var idx0 = offset + (group + j) * stride
                                var idx1 = idx0 + s_stride * stride

                                var w_re = SIMD[Type, simd_width]()
                                var w_im = SIMD[Type, simd_width]()
                                if k >= 4:
                                    if shift == 0:
                                        w_re = twiddle_ptr_re.load[
                                            width=simd_width
                                        ](j)
                                        w_im = twiddle_ptr_im.load[
                                            width=simd_width
                                        ](j)
                                    else:
                                        for v_idx in range(simd_width):
                                            var tidx = (j + v_idx) << shift
                                            w_re[v_idx] = twiddle_ptr_re[tidx]
                                            w_im[v_idx] = twiddle_ptr_im[tidx]
                                else:
                                    for v_idx in range(simd_width):
                                        var w = compute_twiddle[True](
                                            j + v_idx, n_points
                                        )
                                        w_re[v_idx] = w[0]
                                        w_im[v_idx] = w[1]

                                var u_re = ptr_re.load[width=simd_width](idx0)
                                var u_im = ptr_im.load[width=simd_width](idx0)
                                var v_re = ptr_re.load[width=simd_width](idx1)
                                var v_im = ptr_im.load[width=simd_width](idx1)

                                var v_rot_re = v_re * w_re - v_im * w_im
                                var v_rot_im = v_re * w_im + v_im * w_re

                                ptr_re.store(idx0, (u_re + v_rot_re) * scale)
                                ptr_im.store(idx0, (u_im + v_rot_im) * scale)
                                ptr_re.store(idx1, (u_re - v_rot_re) * scale)
                                ptr_im.store(idx1, (u_im - v_rot_im) * scale)

                            vectorize[butterfly_vectorized, simd_width](
                                s_stride
                            )

                        parallelize[group_worker](num_groups)
                    else:
                        # Late stage: few groups, many butterflies per group. Parallelize the J loop.
                        var total_vec_steps = num_groups * (
                            s_stride // simd_width
                        )
                        var j_steps_per_group = s_stride // simd_width

                        @parameter
                        fn global_group_worker(global_vec_idx: Int):
                            var group_idx = global_vec_idx // j_steps_per_group
                            var j = (
                                global_vec_idx % j_steps_per_group
                            ) * simd_width
                            var group = group_idx * s_step

                            var idx0 = offset + (group + j) * stride
                            var idx1 = idx0 + s_stride * stride

                            var w_re = SIMD[Type, simd_width]()
                            var w_im = SIMD[Type, simd_width]()
                            if k >= 4:
                                if shift == 0:
                                    w_re = twiddle_ptr_re.load[
                                        width=simd_width
                                    ](j)
                                    w_im = twiddle_ptr_im.load[
                                        width=simd_width
                                    ](j)
                                else:
                                    for v_idx in range(simd_width):
                                        var tidx = (j + v_idx) << shift
                                        w_re[v_idx] = twiddle_ptr_re[tidx]
                                        w_im[v_idx] = twiddle_ptr_im[tidx]
                            else:
                                for v_idx in range(simd_width):
                                    var w = compute_twiddle[True](
                                        j + v_idx, n_points
                                    )
                                    w_re[v_idx] = w[0]
                                    w_im[v_idx] = w[1]

                            var u_re = ptr_re.load[width=simd_width](idx0)
                            var u_im = ptr_im.load[width=simd_width](idx0)
                            var v_re = ptr_re.load[width=simd_width](idx1)
                            var v_im = ptr_im.load[width=simd_width](idx1)

                            var v_rot_re = v_re * w_re - v_im * w_im
                            var v_rot_im = v_re * w_im + v_im * w_re

                            ptr_re.store(idx0, (u_re + v_rot_re) * scale)
                            ptr_im.store(idx0, (u_im + v_rot_im) * scale)
                            ptr_re.store(idx1, (u_re - v_rot_re) * scale)
                            ptr_im.store(idx1, (u_im - v_rot_im) * scale)

                        parallelize[global_group_worker](total_vec_steps)
        else:
            # Inverse: Parallel DIF Stages
            for stage in reversed(range(k)):
                var s_stride = 1 << stage
                var s_step = s_stride << 1
                var n_points = s_step
                var num_groups = span // s_step
                var shift = k - 1 - stage

                if num_groups >= 16 or s_stride < simd_width:

                    @parameter
                    fn group_worker_inv(group_idx: Int):
                        var group = group_idx * s_step

                        @parameter
                        fn butterfly_vectorized_inv[simd_width: Int](j: Int):
                            var idx0 = offset + (group + j) * stride
                            var idx1 = idx0 + s_stride * stride

                            var w_re = SIMD[Type, simd_width]()
                            var w_im = SIMD[Type, simd_width]()

                            if k >= 4:
                                if shift == 0:
                                    w_re = twiddle_ptr_re.load[
                                        width=simd_width
                                    ](j)
                                    w_im = twiddle_ptr_im.load[
                                        width=simd_width
                                    ](j)
                                else:
                                    for v_idx in range(simd_width):
                                        var tidx = (j + v_idx) << shift
                                        w_re[v_idx] = twiddle_ptr_re[tidx]
                                        w_im[v_idx] = twiddle_ptr_im[tidx]
                            else:
                                for v_idx in range(simd_width):
                                    var w = compute_twiddle[False](
                                        j + v_idx, n_points
                                    )
                                    w_re[v_idx] = w[0]
                                    w_im[v_idx] = w[1]

                            var u_re = ptr_re.load[width=simd_width](idx0)
                            var u_im = ptr_im.load[width=simd_width](idx0)
                            var v_re = ptr_re.load[width=simd_width](idx1)
                            var v_im = ptr_im.load[width=simd_width](idx1)
                            var res_re = (u_re + v_re) * scale
                            var res_im = (u_im + v_im) * scale
                            var diff_re = (u_re - v_re) * scale
                            var diff_im = (u_im - v_im) * scale
                            ptr_re.store(idx0, res_re)
                            ptr_im.store(idx0, res_im)
                            ptr_re.store(idx1, diff_re * w_re - diff_im * w_im)
                            ptr_im.store(idx1, diff_re * w_im + diff_im * w_re)

                        vectorize[butterfly_vectorized_inv, simd_width](
                            s_stride
                        )

                    parallelize[group_worker_inv](num_groups)
                else:
                    # Late stage: few groups, many butterflies per group. Parallelize the J loop.
                    var total_vec_steps = num_groups * (s_stride // simd_width)
                    var j_steps_per_group = s_stride // simd_width

                    @parameter
                    fn global_group_worker_inv(global_vec_idx: Int):
                        var group_idx = global_vec_idx // j_steps_per_group
                        var j = (
                            global_vec_idx % j_steps_per_group
                        ) * simd_width
                        var group = group_idx * s_step

                        var idx0 = offset + (group + j) * stride
                        var idx1 = idx0 + s_stride * stride

                        var w_re = SIMD[Type, simd_width]()
                        var w_im = SIMD[Type, simd_width]()

                        if k >= 4:
                            if shift == 0:
                                w_re = twiddle_ptr_re.load[width=simd_width](j)
                                w_im = twiddle_ptr_im.load[width=simd_width](j)
                            else:
                                for v_idx in range(simd_width):
                                    var tidx = (j + v_idx) << shift
                                    w_re[v_idx] = twiddle_ptr_re[tidx]
                                    w_im[v_idx] = twiddle_ptr_im[tidx]
                        else:
                            for v_idx in range(simd_width):
                                var w = compute_twiddle[False](
                                    j + v_idx, n_points
                                )
                                w_re[v_idx] = w[0]
                                w_im[v_idx] = w[1]

                        var u_re = ptr_re.load[width=simd_width](idx0)
                        var u_im = ptr_im.load[width=simd_width](idx0)
                        var v_re = ptr_re.load[width=simd_width](idx1)
                        var v_im = ptr_im.load[width=simd_width](idx1)
                        var res_re = (u_re + v_re) * scale
                        var res_im = (u_im + v_im) * scale
                        var diff_re = (u_re - v_re) * scale
                        var diff_im = (u_im - v_im) * scale
                        ptr_re.store(idx0, res_re)
                        ptr_im.store(idx0, res_im)
                        ptr_re.store(idx1, diff_re * w_re - diff_im * w_im)
                        ptr_im.store(idx1, diff_re * w_im + diff_im * w_re)

                    parallelize[global_group_worker_inv](total_vec_steps)

            if do_swap:

                @parameter
                fn swap_worker_end(i: Int):
                    var reversed_i = 0
                    for b_idx in range(k):
                        if (i >> b_idx) & 1:
                            reversed_i |= 1 << (k - 1 - b_idx)
                    if i < reversed_i:
                        var idx_i = offset + i * stride
                        var idx_r = offset + reversed_i * stride
                        var tmp_re = ptr_re[idx_i]
                        var tmp_im = ptr_im[idx_i]
                        ptr_re[idx_i] = ptr_re[idx_r]
                        ptr_im[idx_i] = ptr_im[idx_r]
                        ptr_re[idx_r] = tmp_re
                        ptr_im[idx_r] = tmp_im

                parallelize[swap_worker_end](span)
