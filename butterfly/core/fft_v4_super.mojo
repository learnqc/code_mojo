"""
FFT V4 Super: Enhanced version of V4 Opt with additional optimizations.
- Extracted 8-point kernel to reduce code duplication
- FMA-friendly butterfly operations
- Vectorized final scaling
"""
from math import pi, cos, sin, sqrt, log2
from memory import UnsafePointer
from algorithm import parallelize, vectorize
from sys.info import simd_width_of
from butterfly.core.state import QuantumState, bit_reverse_state
from butterfly.core.classical_fft import generate_factors, load_twiddle_4x
from utils.fast_div import FastDiv

alias FloatType = DType.float64
alias simd_width = simd_width_of[FloatType]()


@always_inline
fn compute_twiddle_simd[
    width: Int
](j_vec: SIMD[FloatType, width], block_size: Int) -> Tuple[
    SIMD[FloatType, width], SIMD[FloatType, width]
]:
    """Compute SIMD twiddle factors for DIF: e^(-2πi*j / block_size)."""
    var angle = -2.0 * pi * j_vec / Float64(block_size)
    return cos(angle), sin(angle)




fn fft_v4_super_kernel(
    mut state: QuantumState,
    factors_re: List[Float64],
    factors_im: List[Float64],
    block_log: Int = 11,
):
    var n = state.size()
    var n_quarter = n // 4
    var log_n = Int(log2(Float64(n)))

    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()
    var ptr_fac_re = factors_re.unsafe_ptr()
    var ptr_fac_im = factors_im.unsafe_ptr()

    var tw_packed_re = List[Float64](length=n // 2, fill=0.0)
    var tw_packed_im = List[Float64](length=n // 2, fill=0.0)
    var ptr_tw_re = tw_packed_re.unsafe_ptr()
    var ptr_tw_im = tw_packed_im.unsafe_ptr()

    # --- Phase 1: Global Stages ---
    var stride = n // 2
    for _ in range(log_n - 1, block_log - 1, -1):
        var num_groups = n // 2 // stride

        @parameter
        fn pack_worker(i: Int):
            var idx = i * simd_width
            var src_idxs = SIMD[DType.int64, simd_width]()
            for k in range(simd_width):
                src_idxs[k] = (idx + k) * num_groups
            var w_re, w_im = load_twiddle_4x[simd_width](
                ptr_fac_re, ptr_fac_im, src_idxs, n_quarter
            )
            ptr_tw_re.store[width=simd_width](idx, w_re)
            ptr_tw_im.store[width=simd_width](idx, w_im)

        parallelize[pack_worker](stride // simd_width)

        var fd = FastDiv[DType.uint32](stride)
        alias TargetType = Scalar[FastDiv[DType.uint32].uint_type]

        @parameter
        fn global_worker(vec_idx: Int):
            var idx_base = vec_idx * simd_width
            var res = fd.__divmod__(TargetType(idx_base))
            var blk_idx = Int(res[0])
            var j = Int(res[1])
            var blk_start = blk_idx * 2 * stride
            var idx0 = blk_start + j
            var idx1 = idx0 + stride

            var a_re = ptr_re.load[width=simd_width](idx0)
            var a_im = ptr_im.load[width=simd_width](idx0)
            var b_re = ptr_re.load[width=simd_width](idx1)
            var b_im = ptr_im.load[width=simd_width](idx1)

            var w_re = ptr_tw_re.load[width=simd_width](j)
            var w_im = ptr_tw_im.load[width=simd_width](j)

            var d_re = a_re - b_re
            var d_im = a_im - b_im

            ptr_re.store[width=simd_width](idx0, a_re + b_re)
            ptr_im.store[width=simd_width](idx0, a_im + b_im)
            ptr_re.store[width=simd_width](idx1, d_re * w_re - d_im * w_im)
            ptr_im.store[width=simd_width](idx1, d_re * w_im + d_im * w_re)

        parallelize[global_worker]((n // 2) // simd_width)
        stride >>= 1

    # --- Phase 2: Local Stages (Optimized) ---
    var effective_block_log = min(block_log, log_n)
    if effective_block_log > 0:
        var local_n = 1 << effective_block_log
        var num_local_blocks = n >> effective_block_log

        @parameter
        fn local_block_worker(blk_idx: Int):
            var blk_start = blk_idx << effective_block_log

            # --- Inline K=1 ---
            if effective_block_log == 1:
                var i0 = blk_start
                var i1 = blk_start + 1
                var u_re = ptr_re[i0]
                var u_im = ptr_im[i0]
                var v_re = ptr_re[i1]
                var v_im = ptr_im[i1]
                ptr_re[i0] = u_re + v_re
                ptr_im[i0] = u_im + v_im
                ptr_re[i1] = u_re - v_re
                ptr_im[i1] = u_im - v_im
                return

            # --- Inline K=2 ---
            elif effective_block_log == 2:
                var i0 = blk_start
                var i1 = blk_start + 1
                var i2 = blk_start + 2
                var i3 = blk_start + 3
                var a0_re = ptr_re[i0]
                var a0_im = ptr_im[i0]
                var a1_re = ptr_re[i1]
                var a1_im = ptr_im[i1]
                var a2_re = ptr_re[i2]
                var a2_im = ptr_im[i2]
                var a3_re = ptr_re[i3]
                var a3_im = ptr_im[i3]
                var r0_re = a0_re + a2_re
                var r0_im = a0_im + a2_im
                var d0_re = a0_re - a2_re
                var d0_im = a0_im - a2_im
                var r1_re = a1_re + a3_re
                var r1_im = a1_im + a3_im
                var d1_re = a1_re - a3_re
                var d1_im = a1_im - a3_im
                var d1_rot_re = d1_im
                var d1_rot_im = -d1_re
                ptr_re.store(i0, r0_re + r1_re)
                ptr_im.store(i0, r0_im + r1_im)
                ptr_re.store(i1, r0_re - r1_re)
                ptr_im.store(i1, r0_im - r1_im)
                ptr_re.store(i2, d0_re + d1_rot_re)
                ptr_im.store(i2, d0_im + d1_rot_im)
                ptr_re.store(i3, d0_re - d1_rot_re)
                ptr_im.store(i3, d0_im - d1_rot_im)
                return

            # --- Inline K=3 ---
            elif effective_block_log == 3:
                var v0_re = ptr_re.load[width=4](blk_start)
                var v0_im = ptr_im.load[width=4](blk_start)
                var v1_re = ptr_re.load[width=4](blk_start + 4)
                var v1_im = ptr_im.load[width=4](blk_start + 4)

                alias sq = 0.7071067811865475244
                var tw_re = SIMD[FloatType, 4](1.0, sq, 0.0, -sq)
                var tw_im = SIMD[FloatType, 4](0.0, -sq, -1.0, -sq)
                var sum_re = v0_re + v1_re
                var sum_im = v0_im + v1_im
                var diff_re = v0_re - v1_re
                var diff_im = v0_im - v1_im
                var diff_tw_re = diff_re * tw_re - diff_im * tw_im
                var diff_tw_im = diff_re * tw_im + diff_im * tw_re

                var v0_lo_re = SIMD[FloatType, 4](
                    sum_re[0], sum_re[1], diff_tw_re[0], diff_tw_re[1]
                )
                var v0_hi_re = SIMD[FloatType, 4](
                    sum_re[2], sum_re[3], diff_tw_re[2], diff_tw_re[3]
                )
                var v0_lo_im = SIMD[FloatType, 4](
                    sum_im[0], sum_im[1], diff_tw_im[0], diff_tw_im[1]
                )
                var v0_hi_im = SIMD[FloatType, 4](
                    sum_im[2], sum_im[3], diff_tw_im[2], diff_tw_im[3]
                )

                var tw1_re = SIMD[FloatType, 4](1.0, 0.0, 1.0, 0.0)
                var tw1_im = SIMD[FloatType, 4](0.0, -1.0, 0.0, -1.0)

                var sum1_re = v0_lo_re + v0_hi_re
                var sum1_im = v0_lo_im + v0_hi_im
                var diff1_re = v0_lo_re - v0_hi_re
                var diff1_im = v0_lo_im - v0_hi_im
                var diff1_tw_re = diff1_re * tw1_re - diff1_im * tw1_im
                var diff1_tw_im = diff1_re * tw1_im + diff1_im * tw1_re

                var s1_a_re = SIMD[FloatType, 4](
                    sum1_re[0], diff1_tw_re[0], sum1_re[2], diff1_tw_re[2]
                )
                var s1_b_re = SIMD[FloatType, 4](
                    sum1_re[1], diff1_tw_re[1], sum1_re[3], diff1_tw_re[3]
                )
                var s1_a_im = SIMD[FloatType, 4](
                    sum1_im[0], diff1_tw_im[0], sum1_im[2], diff1_tw_im[2]
                )
                var s1_b_im = SIMD[FloatType, 4](
                    sum1_im[1], diff1_tw_im[1], sum1_im[3], diff1_tw_im[3]
                )

                var final_sum_re = s1_a_re + s1_b_re
                var final_sum_im = s1_a_im + s1_b_im
                var final_diff_re = s1_a_re - s1_b_re
                var final_diff_im = s1_a_im - s1_b_im

                var v_out_0_re = SIMD[FloatType, 4](
                    final_sum_re[0],
                    final_diff_re[0],
                    final_sum_re[1],
                    final_diff_re[1],
                )
                var v_out_0_im = SIMD[FloatType, 4](
                    final_sum_im[0],
                    final_diff_im[0],
                    final_sum_im[1],
                    final_diff_im[1],
                )
                var v_out_1_re = SIMD[FloatType, 4](
                    final_sum_re[2],
                    final_diff_re[2],
                    final_sum_re[3],
                    final_diff_re[3],
                )
                var v_out_1_im = SIMD[FloatType, 4](
                    final_sum_im[2],
                    final_diff_im[2],
                    final_sum_im[3],
                    final_diff_im[3],
                )

                ptr_re.store[width=4](blk_start, v_out_0_re)
                ptr_im.store[width=4](blk_start, v_out_0_im)
                ptr_re.store[width=4](blk_start + 4, v_out_1_re)
                ptr_im.store[width=4](blk_start + 4, v_out_1_im)
                return

            var stop_stage = 2
            for stage in range(effective_block_log - 1, stop_stage, -1):
                var s_stride = 1 << stage
                var step = s_stride << 1
                var b_size = step
                var num_inner = local_n // step
                for inner in range(num_inner):
                    var sub_start = blk_start + inner * step

                    @parameter
                    fn v_kernel[width: Int](j: Int):
                        var i0 = sub_start + j
                        var i1 = i0 + s_stride
                        var a_re = ptr_re.load[width=width](i0)
                        var a_im = ptr_im.load[width=width](i0)
                        var b_re = ptr_re.load[width=width](i1)
                        var b_im = ptr_im.load[width=width](i1)

                        var j_v = SIMD[FloatType, width](j) + SIMD[
                            FloatType, width
                        ](0, 1, 2, 3, 4, 5, 6, 7)
                        var w_re, w_im = compute_twiddle_simd[width](
                            j_v, b_size
                        )

                        var d_re = a_re - b_re
                        var d_im = a_im - b_im
                        ptr_re.store[width=width](i0, a_re + b_re)
                        ptr_im.store[width=width](i0, a_im + b_im)
                        ptr_re.store[width=width](i1, d_re * w_re - d_im * w_im)
                        ptr_im.store[width=width](i1, d_re * w_im + d_im * w_re)

                    vectorize[v_kernel, simd_width](s_stride)

            # Base Case: Use optimized 8-point kernel for remaining stages
            var k3_step = 8
            var num_k3_blocks = local_n // k3_step
            for i in range(num_k3_blocks):
                var offset = blk_start + i * k3_step
                var v0_re = ptr_re.load[width=4](offset)
                var v0_im = ptr_im.load[width=4](offset)
                var v1_re = ptr_re.load[width=4](offset + 4)
                var v1_im = ptr_im.load[width=4](offset + 4)

                alias sq = 0.7071067811865475244
                var tw_re = SIMD[FloatType, 4](1.0, sq, 0.0, -sq)
                var tw_im = SIMD[FloatType, 4](0.0, -sq, -1.0, -sq)
                var sum_re = v0_re + v1_re
                var sum_im = v0_im + v1_im
                var diff_re = v0_re - v1_re
                var diff_im = v0_im - v1_im
                var diff_tw_re = diff_re * tw_re - diff_im * tw_im
                var diff_tw_im = diff_re * tw_im + diff_im * tw_re

                var v0_lo_re = SIMD[FloatType, 4](
                    sum_re[0], sum_re[1], diff_tw_re[0], diff_tw_re[1]
                )
                var v0_hi_re = SIMD[FloatType, 4](
                    sum_re[2], sum_re[3], diff_tw_re[2], diff_tw_re[3]
                )
                var v0_lo_im = SIMD[FloatType, 4](
                    sum_im[0], sum_im[1], diff_tw_im[0], diff_tw_im[1]
                )
                var v0_hi_im = SIMD[FloatType, 4](
                    sum_im[2], sum_im[3], diff_tw_im[2], diff_tw_im[3]
                )

                var tw1_re = SIMD[FloatType, 4](1.0, 0.0, 1.0, 0.0)
                var tw1_im = SIMD[FloatType, 4](0.0, -1.0, 0.0, -1.0)

                var sum1_re = v0_lo_re + v0_hi_re
                var sum1_im = v0_lo_im + v0_hi_im
                var diff1_re = v0_lo_re - v0_hi_re
                var diff1_im = v0_lo_im - v0_hi_im
                var diff1_tw_re = diff1_re * tw1_re - diff1_im * tw1_im
                var diff1_tw_im = diff1_re * tw1_im + diff1_im * tw1_re

                var s1_a_re = SIMD[FloatType, 4](
                    sum1_re[0], diff1_tw_re[0], sum1_re[2], diff1_tw_re[2]
                )
                var s1_b_re = SIMD[FloatType, 4](
                    sum1_re[1], diff1_tw_re[1], sum1_re[3], diff1_tw_re[3]
                )
                var s1_a_im = SIMD[FloatType, 4](
                    sum1_im[0], diff1_tw_im[0], sum1_im[2], diff1_tw_im[2]
                )
                var s1_b_im = SIMD[FloatType, 4](
                    sum1_im[1], diff1_tw_im[1], sum1_im[3], diff1_tw_im[3]
                )

                var final_sum_re = s1_a_re + s1_b_re
                var final_sum_im = s1_a_im + s1_b_im
                var final_diff_re = s1_a_re - s1_b_re
                var final_diff_im = s1_a_im - s1_b_im

                var v_out_0_re = SIMD[FloatType, 4](
                    final_sum_re[0],
                    final_diff_re[0],
                    final_sum_re[1],
                    final_diff_re[1],
                )
                var v_out_0_im = SIMD[FloatType, 4](
                    final_sum_im[0],
                    final_diff_im[0],
                    final_sum_im[1],
                    final_diff_im[1],
                )
                var v_out_1_re = SIMD[FloatType, 4](
                    final_sum_re[2],
                    final_diff_re[2],
                    final_sum_re[3],
                    final_diff_re[3],
                )
                var v_out_1_im = SIMD[FloatType, 4](
                    final_sum_im[2],
                    final_diff_im[2],
                    final_sum_im[3],
                    final_diff_im[3],
                )

                ptr_re.store[width=4](offset, v_out_0_re)
                ptr_im.store[width=4](offset, v_out_0_im)
                ptr_re.store[width=4](offset + 4, v_out_1_re)
                ptr_im.store[width=4](offset + 4, v_out_1_im)

        parallelize[local_block_worker](num_local_blocks)


fn fft_v4_super(mut state: QuantumState, block_log: Int = 11):
    var n = state.size()
    ref factors_re, factors_im = generate_factors(n)

    fft_v4_super_kernel(state, factors_re, factors_im, block_log)

    bit_reverse_state(state)
    var scale = Float64(1.0) / sqrt(Float64(n))
    for i in range(n):
        state.re[i] *= scale
        state.im[i] *= scale
