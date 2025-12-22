"""
FFT V4 Opt: Synthesis of PhastFT, V3, and V6 Kernel Ideas.
Global: Table-based + Twiddle Packing (Phast Trick).
Local: Register Kernels (V6) for small N, Shared Twiddles for large N.
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


fn fft_v4_opt_kernel(
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
                # Optimized scalar logic is fine for 4 points
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

            # --- Inline K=3 Using Vectorization (Width 4) ---
            elif effective_block_log == 3:
                # We process the 8 points as 2 vectors of width 4.
                # v0 = [0,1,2,3], v1 = [4,5,6,7]

                # Load
                var v0_re = ptr_re.load[width=4](blk_start)
                var v0_im = ptr_im.load[width=4](blk_start)
                var v1_re = ptr_re.load[width=4](blk_start + 4)
                var v1_im = ptr_im.load[width=4](blk_start + 4)

                alias sq = 0.7071067811865475244

                # --- Stage 2 (Stride 4) ---
                # Butterfly between v0 and v1. Twiddles for v1 group (indices 4..7).
                # Indices 4,5,6,7 in N=8 correspond to k=4,5,6,7.
                # Twiddles: W_8^0, W_8^1, W_8^2, W_8^3.
                # Consts:
                # 0: 1
                # 1: sq - i*sq
                # 2: -i
                # 3: -sq - i*sq

                var tw_re = SIMD[FloatType, 4](1.0, sq, 0.0, -sq)
                var tw_im = SIMD[FloatType, 4](0.0, -sq, -1.0, -sq)

                var sum_re = v0_re + v1_re
                var sum_im = v0_im + v1_im

                var diff_re = v0_re - v1_re
                var diff_im = v0_im - v1_im

                # Apply twiddles to diff
                var diff_tw_re = diff_re * tw_re - diff_im * tw_im
                var diff_tw_im = diff_re * tw_im + diff_im * tw_re

                # Update current state. Now v0 holds 'sum', v1 holds 'diff_tw'.
                v0_re = sum_re
                v0_im = sum_im
                v1_re = diff_tw_re
                v1_im = diff_tw_im

                # --- Stage 1 (Stride 2) ---
                # Now we have 2 independent DFT-4 problems interleaved?
                # v0 contains inputs for next stage (indices 0,1,2,3 -> results of stage 2).
                # We need butterflies with stride 2.
                # v0: (0,2), (1,3).
                # v1: (4,6), (5,7). (Indices relative to 8-block).

                # Shuffle v0 to align (0,1) with (2,3).
                # slice0 = [0,1], slice1 = [2,3].
                # Mojo SIMD shuffling...
                # We can't easily sub-vectorize.
                # Explicit shuffle:

                # v0 low: (0, 1), high: (2, 3)
                var v0_lo_re = SIMD[FloatType, 4](
                    v0_re[0], v0_re[1], v1_re[0], v1_re[1]
                )
                var v0_hi_re = SIMD[FloatType, 4](
                    v0_re[2], v0_re[3], v1_re[2], v1_re[3]
                )
                var v0_lo_im = SIMD[FloatType, 4](
                    v0_im[0], v0_im[1], v1_im[0], v1_im[1]
                )
                var v0_hi_im = SIMD[FloatType, 4](
                    v0_im[2], v0_im[3], v1_im[2], v1_im[3]
                )

                # Now we butterfly v0_lo and v0_hi
                # Twiddles for hi part (stride 2): W_4^0, W_4^1.
                # W_4^0 = 1, W_4^1 = -i.
                # Pattern repeats for the second block (v1 parts).
                # So we can use vec4 constant: [1, -i, 1, -i]

                var tw1_re = SIMD[FloatType, 4](1.0, 0.0, 1.0, 0.0)
                var tw1_im = SIMD[FloatType, 4](0.0, -1.0, 0.0, -1.0)

                var sum1_re = v0_lo_re + v0_hi_re
                var sum1_im = v0_lo_im + v0_hi_im

                var diff1_re = v0_lo_re - v0_hi_re
                var diff1_im = v0_lo_im - v0_hi_im

                var diff1_tw_re = diff1_re * tw1_re - diff1_im * tw1_im
                var diff1_tw_im = diff1_re * tw1_im + diff1_im * tw1_re

                # Update state:
                # We have results for 0,1,4,5 in sum1.
                # Results for 2,3,6,7 in diff1_tw.
                # Need to repack for Stage 0?
                # Stage 0 is stride 1. (0,1), (2,3)...
                # sum1 has (0,1) and (4,5).
                # diff1_tw has (2,3) and (6,7).
                # We need to butterfly adjacent elements.
                # 0 with 1.
                # sum1[0] vs sum1[1].

                # Repack into:
                # groupA: 0, 2, 4, 6 (Even indices of previous arrays)
                # groupB: 1, 3, 5, 7 (Odd indices)

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

                # Stage 0 (Stride 1)
                # Twiddle is W_2^0 = 1 for all.
                # Just add/sub.

                var final_sum_re = s1_a_re + s1_b_re
                var final_sum_im = s1_a_im + s1_b_im
                var final_diff_re = s1_a_re - s1_b_re
                var final_diff_im = s1_a_im - s1_b_im

                # Now we need to store 0,1,2,3,4,5,6,7 back.
                # final_sum has results for 0, 2, 4, 6?
                # No, wait.
                # A[0] = sum1[0] (idx 0 of local). B[0] = sum1[1] (idx 1).
                # A+B -> 0+1 -> result 0.
                # A-B -> 0-1 -> result 1.
                # So final_sum[0] is result 0. final_diff[0] is result 1.
                # final_sum[1] is derived from inputs 2,3 -> result 2.
                # final_diff[1] is result 3.

                # We need to interleave final_sum and final_diff to store.
                # Output order: s[0], d[0], s[1], d[1], s[2], d[2], s[3], d[3]
                # Corresponds to indices 0,1, 2,3, 4,5, 6,7.

                # Explicit store to avoid complex interleave instructions?
                # Or construct vector?
                # SIMD stores are contiguous.

                # We can store individually? No, scalar stores kill us.
                # Construct interleaved vectors.
                # v_out_0 = [s0, d0, s1, d1]
                # v_out_1 = [s2, d2, s3, d3]

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
                # Generic Loop ...
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

            # Base Case: Dispatch wrapper for remaining stages
            # Handled with generic scalar loop calling optimized vector block
            var k3_step = 8
            var num_k3_blocks = local_n // k3_step
            # We can vectorize THIS loop too?
            # For now, just unroll manually or let compiler do it
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


fn fft_v4_opt(mut state: QuantumState, block_log: Int = 11):
    var n = state.size()
    ref factors_re, factors_im = generate_factors(n)

    fft_v4_opt_kernel(state, factors_re, factors_im, block_log)

    bit_reverse_state(state)
    var scale = Float64(1.0) / sqrt(Float64(n))
    for i in range(n):
        state.re[i] *= scale
        state.im[i] *= scale
