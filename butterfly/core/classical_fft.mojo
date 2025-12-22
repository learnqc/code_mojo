from butterfly.core.state import *
from butterfly.core.types import float_bytes
from butterfly.algos.adjacent_pairs import swap_state_to_distance
from butterfly.algos.tail_stages import (
    stride4_swapped_simd,
)
from butterfly.algos.fused_stages import fused_stride2_stride1_swapped
from butterfly.algos.radix4 import radix4_dif_simd

from memory import UnsafePointer
from algorithm import parallelize, vectorize
from buffer import NDBuffer
from utils.fast_div import FastDiv
from time import perf_counter_ns
from sys.info import simd_width_of
from math import pi, cos, sin, sqrt, log2


fn generate_factors(
    n: Int, inverse: Bool = False
) -> Tuple[List[FloatType], List[FloatType]]:
    var angle = 2.0 * pi / FloatType(n)
    if not inverse:
        angle = -angle

    var wm_re = cos(angle)
    var wm_im = sin(angle)

    alias sqh = sqrt(0.5).cast[Type]()

    # Allocate only N/4 elements (first quadrant)
    factors_re = List[FloatType](length=n // 4, fill=0.0)
    factors_im = List[FloatType](length=n // 4, fill=0.0)

    # Use parallelization for large N
    var limit = n // 8
    alias grain_size = 2048
    var num_chunks = (limit + grain_size - 1) // grain_size

    var ptr_re = factors_re.unsafe_ptr()
    var ptr_im = factors_im.unsafe_ptr()

    @parameter
    fn worker(idx: Int):
        var start = idx * grain_size
        var end = start + grain_size
        if end > limit:
            end = limit

        var thread_angle = angle * FloatType(start)
        var w_re = cos(thread_angle)
        var w_im = sin(thread_angle)

        for j in range(start, end):
            # Store Quadrant 1 (0..N/8)
            ptr_re[j] = w_re
            ptr_im[j] = w_im

            # Store Octant 2 (N/8..N/4) using symmetry
            # W^{N/8 + j} = ((w_re + w_im) - i(w_re - w_im)) * 1/sqrt(2)
            var sum = (w_re + w_im) * sqh
            var diff = (w_re - w_im) * sqh

            ptr_re[j + n // 8] = sum
            ptr_im[j + n // 8] = -diff

            # Update w = w * wm
            var t_re = w_re * wm_re - w_im * wm_im
            var t_im = w_re * wm_im + w_im * wm_re
            w_re = t_re
            w_im = t_im

    parallelize[worker](num_chunks)

    return factors_re^, factors_im^


fn fft_dit(mut state: QuantumState, inverse: Bool = False):
    """Apply FFT to the quantum state using Cooley-Tukey algorithm.

    This is an in-place FFT that operates on the state's complex amplitudes.
    Uses the radix-2 decimation-in-time algorithm.

    Args:
        state: QuantumState to transform (modified in-place)
        inverse: If True, performs inverse FFT
    """
    var n = state.size()

    # Bit-reversal permutation
    bit_reverse_state(state)

    ref factors_re, factors_im = generate_factors(n)

    # Cooley-Tukey FFT
    var m = 2
    while m <= n:
        f = n // m
        # Process all groups at this stage
        for k in range(0, n, m):
            # Butterfly operations within group
            for j in range(m // 2):
                w_re = factors_re[f * j]
                w_im = factors_im[f * j]
                var t_re = (
                    w_re * state.re[k + j + m // 2]
                    - w_im * state.im[k + j + m // 2]
                )
                var t_im = (
                    w_re * state.im[k + j + m // 2]
                    + w_im * state.re[k + j + m // 2]
                )

                var u_re = state.re[k + j]
                var u_im = state.im[k + j]

                state.re[k + j] = u_re + t_re
                state.im[k + j] = u_im + t_im
                state.re[k + j + m // 2] = u_re - t_re
                state.im[k + j + m // 2] = u_im - t_im

        m *= 2

    var scale = FloatType(1.0) / sqrt(FloatType(n))
    for i in range(n):
        state.re[i] *= scale
        state.im[i] *= scale


fn fft_dif(mut state: QuantumState, inverse: Bool = False):
    """NumPy-style FFT using decimation-in-frequency algorithm."""
    var n = state.size()
    var log_n = Int(log2(Float64(n)))

    # Factors generation (reused or recomputed)
    # We need to inline this properly or share it.
    # For now, we reuse the logic but we need access to factors.
    # The original fft_dif computed factors inside? No, it used generate_factors equivalent logic implicitly?
    # Actually the previous file had a manual copy of factor generation inside generate_factors AND inside fft_dif factors loop.
    # Let's just use generate_factors for cleaner code, assuming cost is negligible or we hoist it.

    ref factors_re, factors_im = generate_factors(n)

    var block_size = n

    for _ in range(log_n):
        var half_block = block_size // 2
        var _ = n // block_size  # Stride for factors?
        # In DIF, we need w^(k * N/block_size)?
        # Original code used: factors_re[n // block_size * j]
        # So stride is n // block_size.

        var stride = n // block_size

        for block_start in range(0, n, block_size):
            for j in range(half_block):
                var top_idx = block_start + j
                var bot_idx = top_idx + half_block

                var top_re = state.re[top_idx]
                var top_im = state.im[top_idx]
                var bot_re = state.re[bot_idx]
                var bot_im = state.im[bot_idx]

                var sum_re = top_re + bot_re
                var sum_im = top_im + bot_im
                var diff_re = top_re - bot_re
                var diff_im = top_im - bot_im

                var w_re = factors_re[stride * j]
                var w_im = factors_im[stride * j]

                var t_re = w_re * diff_re - w_im * diff_im
                var t_im = w_re * diff_im + w_im * diff_re

                # In DIF, we multiply AFTER butterfly on the bottom part?
                # Standard DIF:
                # a = u + v
                # b = (u - v) * w
                # Yes, matching logic above.

                state.re[top_idx] = sum_re
                state.im[top_idx] = sum_im
                state.re[bot_idx] = t_re
                state.im[bot_idx] = t_im

        block_size = half_block

    bit_reverse_state(state)

    var scale = FloatType(1.0) / sqrt(FloatType(n))
    for i in range(n):
        state.re[i] *= scale
        state.im[i] *= scale


fn fft_dif_simd(mut state: QuantumState, inverse: Bool = False):
    """Vectorized DIF FFT."""
    var n = state.size()
    var log_n = Int(log2(Float64(n)))
    ref factors_re, factors_im = generate_factors(n)

    var ptr_re = UnsafePointer(state.re.unsafe_ptr())
    var ptr_im = UnsafePointer(state.im.unsafe_ptr())

    var block_size = n
    for _ in range(log_n):
        var half_block = block_size // 2
        var stride = n // block_size

        for block_start in range(0, n, block_size):
            if half_block >= simd_width:

                @parameter
                fn inner_simd[width: Int](j: Int):
                    var top_idx = block_start + j
                    var bot_idx = top_idx + half_block

                    var top_re = ptr_re.load[width=width](top_idx)
                    var top_im = ptr_im.load[width=width](top_idx)
                    var bot_re = ptr_re.load[width=width](bot_idx)
                    var bot_im = ptr_im.load[width=width](bot_idx)

                    var sum_re = top_re + bot_re
                    var sum_im = top_im + bot_im
                    var diff_re = top_re - bot_re
                    var diff_im = top_im - bot_im

                    # Factors
                    var w_re: SIMD[Type, width]
                    var w_im: SIMD[Type, width]

                    if stride == 1:
                        w_re = factors_re.unsafe_ptr().load[width=width](j)
                        w_im = factors_im.unsafe_ptr().load[width=width](j)
                    else:
                        # Manual gather for strided access
                        w_re = SIMD[Type, width]()
                        w_im = SIMD[Type, width]()
                        for k in range(width):
                            w_re[k] = factors_re.unsafe_ptr().load(
                                stride * (j + k)
                            )
                            w_im[k] = factors_im.unsafe_ptr().load(
                                stride * (j + k)
                            )

                    # t = diff * w
                    var t_re = diff_re * w_re - diff_im * w_im
                    var t_im = diff_re * w_im + diff_im * w_re

                    ptr_re.store[width=width](top_idx, sum_re)
                    ptr_im.store[width=width](top_idx, sum_im)
                    ptr_re.store[width=width](bot_idx, t_re)
                    ptr_im.store[width=width](bot_idx, t_im)

                vectorize[inner_simd, simd_width](half_block)
            else:
                # Scalar fallback
                for j in range(half_block):
                    var w_re = factors_re[stride * j]
                    var w_im = factors_im[stride * j]

                    var top_idx = block_start + j
                    var bot_idx = top_idx + half_block

                    var top_re = state.re[top_idx]
                    var top_im = state.im[top_idx]
                    var bot_re = state.re[bot_idx]
                    var bot_im = state.im[bot_idx]

                    var sum_re = top_re + bot_re
                    var sum_im = top_im + bot_im
                    var diff_re = top_re - bot_re
                    var diff_im = top_im - bot_im

                    var t_re = diff_re * w_re - diff_im * w_im
                    var t_im = diff_re * w_im + diff_im * w_re

                    state.re[top_idx] = sum_re
                    state.im[top_idx] = sum_im
                    state.re[bot_idx] = t_re
                    state.im[bot_idx] = t_im

        block_size = half_block

    bit_reverse_state(state)

    var scale = FloatType(1.0) / sqrt(FloatType(n))
    var ptr_re_norm = UnsafePointer(state.re.unsafe_ptr())
    var ptr_im_norm = UnsafePointer(state.im.unsafe_ptr())

    @parameter
    fn norm_simd[width: Int](i: Int):
        ptr_re_norm.store[width=width](
            i, ptr_re_norm.load[width=width](i) * scale
        )
        ptr_im_norm.store[width=width](
            i, ptr_im_norm.load[width=width](i) * scale
        )

    vectorize[norm_simd, simd_width](n)


@always_inline
fn load_twiddle_4x[
    width: Int
](
    ptr_fac_re: UnsafePointer[FloatType],
    ptr_fac_im: UnsafePointer[FloatType],
    k: SIMD[DType.int64, width],
    n_quarter: Int,
) -> Tuple[SIMD[DType.float64, width], SIMD[DType.float64, width]]:
    """Derive twiddle factors for second quadrant on the fly.
    Identity: W^{j+N/4} = Im(W^j) - i Re(W^j).
    """

    @parameter
    if width > 1:
        alias bool_vec = SIMD[DType.bool, width]
        var is_q2_bool = rebind[bool_vec](k.ge(n_quarter))
        var is_q2 = is_q2_bool.cast[DType.float64]()
        var is_q1 = 1.0 - is_q2

        var k_adj = k - is_q2_bool.cast[DType.int64]() * n_quarter
        var w_re_base = ptr_fac_re.gather(k_adj)
        var w_im_base = ptr_fac_im.gather(k_adj)

        var w_re = is_q1 * w_re_base + is_q2 * w_im_base
        var w_im = is_q1 * w_im_base - is_q2 * w_re_base
        return w_re, w_im
    else:
        var k_val = Int(k[0])
        if k_val >= n_quarter:
            var k_adj = k_val - n_quarter
            return ptr_fac_im[k_adj], -ptr_fac_re[k_adj]
        else:
            return ptr_fac_re[k_val], ptr_fac_im[k_val]


fn fft_dif_parallel_simd_phast_kernel(
    mut state: QuantumState,
    factors_re: List[FloatType],
    factors_im: List[FloatType],
):
    """Core butterfly kernel for Phast FFT."""
    var n = state.size()
    var n_quarter = n // 4
    var log_n = Int(log2(Float64(n)))

    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()
    var ptr_fac_re = factors_re.unsafe_ptr()
    var ptr_fac_im = factors_im.unsafe_ptr()

    var stride = n // 2

    # Pre-allocate a scratch buffer for twiddles using List
    var tw_max = n // 2
    var tw_buf_re = List[FloatType](capacity=tw_max)
    var tw_buf_im = List[FloatType](capacity=tw_max)

    var ptr_tw_buf_re = tw_buf_re.unsafe_ptr()
    var ptr_tw_buf_im = tw_buf_im.unsafe_ptr()

    for _ in range(log_n):
        var fd = FastDiv[DType.uint32](stride)
        alias TargetType = Scalar[FastDiv[DType.uint32].uint_type]
        var num_groups = n // 2 // stride

        if (stride // 2) >= simd_width:
            alias target_l2_bytes = 4 * 1024 * 1024
            alias target_l1_bytes = 64 * 1024
            alias complex_size = 2 * float_bytes
            alias TILED_THRESHOLD = target_l2_bytes // complex_size
            alias TILE_SIZE = target_l1_bytes // complex_size

            if num_groups <= 4 and stride >= TILED_THRESHOLD:
                # Tiled Zero-Copy Mode
                var num_items = n // 2
                var num_tiles = num_items // TILE_SIZE

                @parameter
                fn worker_tiled(tile_idx: Int):
                    var base_idx = tile_idx * TILE_SIZE
                    for k in range(0, TILE_SIZE, simd_width):
                        var idx_base = base_idx + k
                        var res = fd.__divmod__(TargetType(idx_base))
                        var block_idx = Int(res[0])
                        var j_base = Int(res[1])
                        var block_start = block_idx * 2 * stride
                        var top_idx_base = block_start + j_base
                        var bot_idx_base = top_idx_base + stride

                        var top_idxs = SIMD[DType.int64, simd_width]()
                        var bot_idxs = SIMD[DType.int64, simd_width]()
                        var fac_idxs = SIMD[DType.int64, simd_width]()

                        for i in range(simd_width):
                            top_idxs[i] = Int64(top_idx_base + i)
                            bot_idxs[i] = Int64(bot_idx_base + i)
                            fac_idxs[i] = Int64((j_base + i) * num_groups)

                        var w_re, w_im = load_twiddle_4x[simd_width](
                            ptr_fac_re, ptr_fac_im, fac_idxs, n_quarter
                        )
                        var top_re = ptr_re.gather(top_idxs)
                        var top_im = ptr_im.gather(top_idxs)
                        var bot_re = ptr_re.gather(bot_idxs)
                        var bot_im = ptr_im.gather(bot_idxs)

                        var sum_re = top_re + bot_re
                        var sum_im = top_im + bot_im
                        var diff_re = top_re - bot_re
                        var diff_im = top_im - bot_im

                        var t_re = diff_re * w_re - diff_im * w_im
                        var t_im = diff_re * w_im + diff_im * w_re

                        ptr_re.scatter(top_idxs, sum_re)
                        ptr_im.scatter(top_idxs, sum_im)
                        ptr_re.scatter(bot_idxs, t_re)
                        ptr_im.scatter(bot_idxs, t_im)

                parallelize[worker_tiled](num_tiles)
            elif num_groups == 1:

                @parameter
                fn worker_fac(vec_idx: Int):
                    var idx_base = vec_idx * simd_width
                    var res = fd.__divmod__(TargetType(idx_base))
                    var block_idx = Int(res[0])
                    var j_base = Int(res[1])
                    var block_start = block_idx * 2 * stride
                    var top_idx_base = block_start + j_base
                    var bot_idx_base = top_idx_base + stride
                    var top_idxs = SIMD[DType.int64, simd_width]()
                    var bot_idxs = SIMD[DType.int64, simd_width]()
                    for i in range(simd_width):
                        top_idxs[i] = Int64(top_idx_base + i)
                        bot_idxs[i] = Int64(bot_idx_base + i)
                    var k_base = j_base * num_groups
                    var k_vec = SIMD[DType.int64, simd_width]()
                    for i in range(simd_width):
                        k_vec[i] = k_base + i
                    var w_re, w_im = load_twiddle_4x[simd_width](
                        ptr_fac_re, ptr_fac_im, k_vec, n_quarter
                    )
                    var top_re = ptr_re.gather(top_idxs)
                    var top_im = ptr_im.gather(top_idxs)
                    var bot_re = ptr_re.gather(bot_idxs)
                    var bot_im = ptr_im.gather(bot_idxs)
                    var sum_re = top_re + bot_re
                    var sum_im = top_im + bot_im
                    var diff_re = top_re - bot_re
                    var diff_im = top_im - bot_im
                    var t_re = diff_re * w_re - diff_im * w_im
                    var t_im = diff_re * w_im + diff_im * w_re
                    ptr_re.scatter(top_idxs, sum_re)
                    ptr_im.scatter(top_idxs, sum_im)
                    ptr_re.scatter(bot_idxs, t_re)
                    ptr_im.scatter(bot_idxs, t_im)

                parallelize[worker_fac]((n // 2) // simd_width, 8)
            else:
                for idx in range(0, stride, simd_width):
                    var k_vec = SIMD[DType.int64, simd_width]()
                    for i in range(simd_width):
                        k_vec[i] = Int64((idx + i) * num_groups)
                    var val_re, val_im = load_twiddle_4x[simd_width](
                        ptr_fac_re, ptr_fac_im, k_vec, n_quarter
                    )
                    ptr_tw_buf_re.store(idx, val_re)
                    ptr_tw_buf_im.store(idx, val_im)

                @parameter
                fn worker_buf(vec_idx: Int):
                    var idx_base = vec_idx * simd_width
                    var res = fd.__divmod__(TargetType(idx_base))
                    var block_idx = Int(res[0])
                    var j_base = Int(res[1])
                    var block_start = block_idx * 2 * stride
                    var top_idx_base = block_start + j_base
                    var bot_idx_base = top_idx_base + stride
                    var top_idxs = SIMD[DType.int64, simd_width]()
                    var bot_idxs = SIMD[DType.int64, simd_width]()
                    for i in range(simd_width):
                        top_idxs[i] = Int64(top_idx_base + i)
                        bot_idxs[i] = Int64(bot_idx_base + i)
                    var w_re = ptr_tw_buf_re.load[width=simd_width](j_base)
                    var w_im = ptr_tw_buf_im.load[width=simd_width](j_base)
                    var top_re = ptr_re.gather(top_idxs)
                    var top_im = ptr_im.gather(top_idxs)
                    var bot_re = ptr_re.gather(bot_idxs)
                    var bot_im = ptr_im.gather(bot_idxs)
                    var sum_re = top_re + bot_re
                    var sum_im = top_im + bot_im
                    var diff_re = top_re - bot_re
                    var diff_im = top_im - bot_im
                    var t_re = diff_re * w_re - diff_im * w_im
                    var t_im = diff_re * w_im + diff_im * w_re
                    ptr_re.scatter(top_idxs, sum_re)
                    ptr_im.scatter(top_idxs, sum_im)
                    ptr_re.scatter(bot_idxs, t_re)
                    ptr_im.scatter(bot_idxs, t_im)

                parallelize[worker_buf]((n // 2) // simd_width, 8)
        else:
            for block_idx in range(n // (2 * stride)):
                var block_start = block_idx * 2 * stride
                for j in range(stride):
                    var idx0 = block_start + j
                    var idx1 = idx0 + stride
                    var k = j * num_groups
                    var w_re: FloatType
                    var w_im: FloatType
                    if k >= n_quarter:
                        w_re = ptr_fac_im[k - n_quarter]
                        w_im = -ptr_fac_re[k - n_quarter]
                    else:
                        w_re = ptr_fac_re[k]
                        w_im = ptr_fac_im[k]
                    var a_re = state.re[idx0]
                    var a_im = state.im[idx0]
                    var b_re = state.re[idx1]
                    var b_im = state.im[idx1]
                    var diff_re = a_re - b_re
                    var diff_im = a_im - b_im
                    state.re[idx0] = a_re + b_re
                    state.im[idx0] = a_im + b_im
                    state.re[idx1] = diff_re * w_re - diff_im * w_im
                    state.im[idx1] = diff_re * w_im + diff_im * w_re
        stride >>= 1


fn fft_dif_parallel_simd_phast(
    mut state: QuantumState, inverse: Bool = False, debug_time: Bool = False
):
    """Parallelized + Vectorized DIF FFT using sequential twiddle buffering (PhastFT style).
    """
    var n = state.size()
    ref factors_re, factors_im = generate_factors(n, inverse)

    fft_dif_parallel_simd_phast_kernel(state, factors_re, factors_im)

    bit_reverse_state(state)
    var scale = 1.0 / sqrt(Float64(n)).cast[Type]()
    for i in range(n):
        state.re[i] *= scale
        state.im[i] *= scale


fn fft_dif_parallel_simd_fused(
    mut state: QuantumState, inverse: Bool = False, debug_time: Bool = False
):
    """Parallelized + Vectorized DIF FFT using Fused Kernels.
    Uses 'fused_stage0_swap_simd' for Stride N/2.
    Uses 'fused_restore_order_simd' for Stride 4 (Restoring N/2 swap).
    """
    var n = state.size()
    var log_n = Int(log2(Float64(n)))
    ref factors_re, factors_im = generate_factors(n)

    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()
    var ptr_fac_re = factors_re.unsafe_ptr()
    var ptr_fac_im = factors_im.unsafe_ptr()

    var stride = n // 2

    # Pre-allocate twiddle buffer
    var tw_max = n // 2
    var tw_buf_re = List[FloatType](capacity=tw_max)
    var tw_buf_im = List[FloatType](capacity=tw_max)
    var ptr_tw_buf_re = tw_buf_re.unsafe_ptr()
    var ptr_tw_buf_im = tw_buf_im.unsafe_ptr()

    var t_start = perf_counter_ns()

    # Track swap state. target_bit (N/2) <-> 2 (Val 4)
    var swapped = False
    var swap_target_bit = log_n - 1

    for _ in range(log_n):
        var t_stage_start = perf_counter_ns()
        var fd = FastDiv[DType.uint32](stride)
        alias TargetType = Scalar[FastDiv[DType.uint32].uint_type]

        var num_groups = n // 2 // stride
        # print("DEBUG: Processing Stride", stride)

        # Check for Fused Opportunities
        if stride == n // 2 and n >= 16:
            # Stage 0: Fused Swap
            from butterfly.algos.fused_stages import fused_stage0_swap_simd

            # print("DEBUG: Executing Fused Stage 0")
            fused_stage0_swap_simd(
                state, factors_re, factors_im, swap_target_bit
            )
            # print("Finished Stage 0")
            swapped = True
            if debug_time:
                # print(
                #     "Stage 0 (Fused Swap): ",
                #     (perf_counter_ns() - t_stage_start) / 1e6,
                #     "ms",
                # )
                pass  # Keep pass to maintain indentation

            stride = stride // 2
            continue

        elif stride == 4 and swapped:
            # Stage 4 + 2 + 1: Fused Restore & Tail
            from butterfly.algos.fused_stages import (
                fused_stage0_swap_simd,
                fused_restore_order_simd,
                fused_stride2_stride1_swapped,
                fused_restore_and_tail_simd,
            )
            from butterfly.algos.radix4 import radix4_dif_simd

            # print("DEBUG: Executing Fused Restore order")
            fused_restore_and_tail_simd(
                state, factors_re, factors_im, swap_target_bit
            )

            # Since Stride 2 and 1 are fused inside, we are done.
            break

        elif (stride // 2) >= simd_width:
            # Standard Tiled/Phast Logic

            # If swapped is True, we need to handle the fact that Bit 4 (Logical) is at P_N/2 (Physical).
            # And Bit N/2 (Logical) is at P_4 (Physical).
            # Stride K (Physical) corresponds to Stride K (Logical) because K != N/2, 4.
            # Twiddles depend on Logical j.
            # Logical j has bits 0..logK-1.
            # If K <= 4, then bits 0..logK-1 do NOT include Bit 4?
            # No, if Stride K=4 (Stage 4), bits are 0..1. Don't include Bit 2(Val 4). Correct.
            # So only Strides > 4 are affected by the swap in terms of Twiddle Index.
            # For Stride > 4, we must adjust 'j'.

            # Compute tiling thresholds based on Cache Sizes
            alias target_l2_bytes = 4 * 1024 * 1024
            alias target_l1_bytes = 64 * 1024
            alias complex_size = 2 * float_bytes

            alias TILED_THRESHOLD = target_l2_bytes // complex_size
            alias TILE_SIZE = target_l1_bytes // complex_size

            if not swapped and num_groups <= 4 and stride >= TILED_THRESHOLD:
                # Zero Copy Tiled (Standard)
                var num_items = n // 2
                var num_tiles = num_items // TILE_SIZE

                @parameter
                fn worker_tiled(tile_idx: Int):
                    var base_idx = tile_idx * TILE_SIZE
                    for k in range(0, TILE_SIZE, simd_width):
                        var idx_base = base_idx + k
                        var res = fd.__divmod__(TargetType(idx_base))
                        var block_idx = Int(res[0])
                        var j_base = Int(res[1])
                        var block_start = block_idx * 2 * stride
                        var top_idx_base = block_start + j_base
                        var bot_idx_base = top_idx_base + stride

                        var top_idxs = SIMD[DType.int64, simd_width]()
                        var bot_idxs = SIMD[DType.int64, simd_width]()
                        var fac_idxs = SIMD[DType.int64, simd_width]()

                        for i in range(simd_width):
                            top_idxs[i] = Int64(top_idx_base + i)
                            bot_idxs[i] = Int64(bot_idx_base + i)
                            fac_idxs[i] = Int64((j_base + i) * num_groups)

                        var w_re = ptr_fac_re.gather(fac_idxs)
                        var w_im = ptr_fac_im.gather(fac_idxs)

                        var top_re = ptr_re.gather(top_idxs)
                        var top_im = ptr_im.gather(top_idxs)
                        var bot_re = ptr_re.gather(bot_idxs)
                        var bot_im = ptr_im.gather(bot_idxs)

                        var sum_re = top_re + bot_re
                        var sum_im = top_im + bot_im
                        var diff_re = top_re - bot_re
                        var diff_im = top_im - bot_im

                        var t_re = diff_re * w_re - diff_im * w_im
                        var t_im = diff_re * w_im + diff_im * w_re

                        ptr_re.scatter(top_idxs, sum_re)
                        ptr_im.scatter(top_idxs, sum_im)
                        ptr_re.scatter(bot_idxs, t_re)
                        ptr_im.scatter(bot_idxs, t_im)

                parallelize[worker_tiled](num_tiles)

            elif swapped and stride > 4:
                # Swapped Mode Tiled Logic (No Phast Buffer)
                # We construct Logical j from Physical j and Block info.
                var num_items = n // 2
                var num_tiles = num_items // TILE_SIZE

                # Precompute bit masks
                # Swap Bit: swap_target_bit (N/2) and 2 (Val 4).
                var bit_n2_shift = swap_target_bit

                @parameter
                fn worker_tiled_swapped(tile_idx: Int):
                    var base_idx = tile_idx * TILE_SIZE
                    for k in range(0, TILE_SIZE, simd_width):
                        var idx_base = base_idx + k
                        var res = fd.__divmod__(TargetType(idx_base))
                        var block_idx = Int(res[0])
                        var j_base = Int(res[1])
                        var block_start = block_idx * 2 * stride
                        var top_idx_base = block_start + j_base
                        var bot_idx_base = top_idx_base + stride

                        var fac_idxs = SIMD[DType.int64, simd_width]()
                        var top_idxs = SIMD[DType.int64, simd_width]()
                        var bot_idxs = SIMD[DType.int64, simd_width]()

                        for i in range(simd_width):
                            top_idxs[i] = Int64(top_idx_base + i)
                            bot_idxs[i] = Int64(bot_idx_base + i)

                            # Twiddle Correction
                            var p_idx = top_idx_base + i
                            var bit_pn2 = (p_idx >> bit_n2_shift) & 1
                            var j_phys = j_base + i
                            # j_log = (j_phys & ~4) | (bit_pn2 << 2)
                            var j_log = (j_phys & ~4) | (bit_pn2 << 2)

                            fac_idxs[i] = Int64(j_log * num_groups)

                        var w_re = ptr_fac_re.gather(fac_idxs)
                        var w_im = ptr_fac_im.gather(fac_idxs)

                        var top_re = ptr_re.gather(top_idxs)
                        var top_im = ptr_im.gather(top_idxs)
                        var bot_re = ptr_re.gather(bot_idxs)
                        var bot_im = ptr_im.gather(bot_idxs)

                        var sum_re = top_re + bot_re
                        var sum_im = top_im + bot_im
                        var diff_re = top_re - bot_re
                        var diff_im = top_im - bot_im

                        var t_re = diff_re * w_re - diff_im * w_im
                        var t_im = diff_re * w_im + diff_im * w_re

                        ptr_re.scatter(top_idxs, sum_re)
                        ptr_im.scatter(top_idxs, sum_im)
                        ptr_re.scatter(bot_idxs, t_re)
                        ptr_im.scatter(bot_idxs, t_im)

                parallelize[worker_tiled_swapped](num_tiles)

        elif (stride // 2) >= simd_width:
            # Standard Phast Buffering Logic for unswapped state
            # Pack first
            for idx in range(0, stride, simd_width):
                var idx_base = idx
                var src_idxs = SIMD[DType.int64, simd_width]()
                for i in range(simd_width):
                    src_idxs[i] = Int64((idx_base + i) * num_groups)
                var val_re = ptr_fac_re.gather(src_idxs)
                var val_im = ptr_fac_im.gather(src_idxs)
                ptr_tw_buf_re.store(idx_base, val_re)
                ptr_tw_buf_im.store(idx_base, val_im)

            # Run using buffer
            @parameter
            fn worker_buf(vec_idx: Int):
                var idx_base = vec_idx * simd_width
                var res = fd.__divmod__(TargetType(idx_base))
                var block_idx = Int(res[0])
                var j_base = Int(res[1])
                var block_start = block_idx * 2 * stride
                var top_idx_base = block_start + j_base
                var bot_idx_base = top_idx_base + stride

                var top_idxs = SIMD[DType.int64, simd_width]()
                var bot_idxs = SIMD[DType.int64, simd_width]()
                for i in range(simd_width):
                    top_idxs[i] = Int64(top_idx_base + i)
                    bot_idxs[i] = Int64(bot_idx_base + i)

                var w_re = ptr_tw_buf_re.load[width=simd_width](j_base)
                var w_im = ptr_tw_buf_im.load[width=simd_width](j_base)

                var top_re = ptr_re.gather(top_idxs)
                var top_im = ptr_im.gather(top_idxs)
                var bot_re = ptr_re.gather(bot_idxs)
                var bot_im = ptr_im.gather(bot_idxs)

                var sum_re = top_re + bot_re
                var sum_im = top_im + bot_im
                var diff_re = top_re - bot_re
                var diff_im = top_im - bot_im

                var t_re = diff_re * w_re - diff_im * w_im
                var t_im = diff_re * w_im + diff_im * w_re

                ptr_re.scatter(top_idxs, sum_re)
                ptr_im.scatter(top_idxs, sum_im)
                ptr_re.scatter(bot_idxs, t_re)
                ptr_im.scatter(bot_idxs, t_im)

            parallelize[worker_buf]((n // 2) // simd_width, 8)

            # Manual Continue to skip Fallback logic
            stride = stride // 2
            continue
        elif stride == 2 and n >= 8:
            # Stride 2 + 1 Fused (Optimized Tail)
            # Swap Bit 1 (Val 2) to Bit 2 (Val 4/Distance 4) to enable vectorization width 4.
            swap_state_to_distance(state, 1, 4)
            fused_stride2_stride1_swapped(state)
            swap_state_to_distance(state, 1, 4)
            break
        else:
            # Fallback (Small Strides or Scalar)
            # (Optimized kernels disabled to ensure correctness)

            # Scalar Fallback
            for idx in range(n // 2):
                var res = fd.__divmod__(TargetType(idx))
                var block_idx = Int(res[0])
                var j = Int(res[1])
                if swapped and stride > 4:
                    # Correct twiddle j
                    var p_N2 = (
                        block_idx * 2 * stride + j
                    ) >> swap_target_bit & 1
                    j = (j & ~16) | (p_N2 << 4)

                var block_start = block_idx * 2 * stride
                var top_idx = block_start + j
                var bot_idx = top_idx + stride

                var w_re = ptr_fac_re[j * num_groups]
                var w_im = ptr_fac_im[j * num_groups]

                var top_re = ptr_re[top_idx]
                var top_im = ptr_im[top_idx]
                var bot_re = ptr_re[bot_idx]
                var bot_im = ptr_im[bot_idx]

                var sum_re = top_re + bot_re
                var sum_im = top_im + bot_im
                var diff_re = top_re - bot_re
                var diff_im = top_im - bot_im

                var t_re = diff_re * w_re - diff_im * w_im
                var t_im = diff_re * w_im + diff_im * w_re

                ptr_re[top_idx] = sum_re
                ptr_im[top_idx] = sum_im
                ptr_re[bot_idx] = t_re
                ptr_im[bot_idx] = t_im

        if debug_time:
            var t_now = perf_counter_ns()
            print(
                "Stage Stride",
                stride,
                ": ",
                (t_now - t_stage_start) / 1e6,
                "ms",
            )

        stride = stride // 2

    if debug_time:
        print("Total Stages: ", (perf_counter_ns() - t_start) / 1e6, "ms")

    bit_reverse_state(state)

    # Normalization
    var scale = FloatType(1.0) / sqrt(FloatType(n))
    var final_ptr_re = state.re.unsafe_ptr()
    var final_ptr_im = state.im.unsafe_ptr()

    for i in range(n):
        final_ptr_re[i] *= scale
        final_ptr_im[i] *= scale


fn fft_radix4_dif(
    mut state: QuantumState,
    factors_re: List[FloatType],
    factors_im: List[FloatType],
):
    """
    Radix-4 DIF FFT.
    Fuses pairs of Radix-2 stages into Radix-4 stages.
    Handles odd power-of-2 sizes by running one initial Radix-2 stage.
    """
    var n = state.size()
    var log_n = Int(log2(FloatType(n)))

    var current_n = n

    # Handle Odd Stage
    if log_n % 2 != 0:
        # Run standard Radix-2 Stage (Stride N/2)
        var stride = n // 2
        var ptr_re = state.re.unsafe_ptr()
        var ptr_im = state.im.unsafe_ptr()
        var ptr_fac_re = factors_re.unsafe_ptr()
        var ptr_fac_im = factors_im.unsafe_ptr()

        alias simd_width = simd_width_of[Type]()

        # Inline loop for simplicity (scalar/vectorized loop)
        for k in range(0, stride, simd_width):
            var vl = simd_width
            if k + vl > stride:
                vl = stride - k

            if vl < simd_width:
                # Scalar tail
                for i in range(vl):
                    var ki = k + i
                    var re0 = ptr_re[ki]
                    var im0 = ptr_im[ki]
                    var re1 = ptr_re[ki + stride]
                    var im1 = ptr_im[ki + stride]

                    var sum_re = re0 + re1
                    var sum_im = im0 + im1
                    var diff_re = re0 - re1
                    var diff_im = im0 - im1

                    var w_re = ptr_fac_re[ki]
                    var w_im = ptr_fac_im[ki]

                    var t_re = diff_re * w_re - diff_im * w_im
                    var t_im = diff_re * w_im + diff_im * w_re

                    ptr_re[ki] = sum_re
                    ptr_im[ki] = sum_im
                    ptr_re[ki + stride] = t_re
                    ptr_im[ki + stride] = t_im
            else:
                # Vector
                var re0 = ptr_re.load[width=simd_width](k)
                var im0 = ptr_im.load[width=simd_width](k)
                var re1 = ptr_re.load[width=simd_width](k + stride)
                var im1 = ptr_im.load[width=simd_width](k + stride)

                var sum_re = re0 + re1
                var sum_im = im0 + im1
                var diff_re = re0 - re1
                var diff_im = im0 - im1

                var w_re = ptr_fac_re.load[width=simd_width](k)
                var w_im = ptr_fac_im.load[width=simd_width](k)

                var t_re = diff_re * w_re - diff_im * w_im
                var t_im = diff_re * w_im + diff_im * w_re

                ptr_re.store(k, sum_re)
                ptr_im.store(k, sum_im)
                ptr_re.store(k + stride, t_re)
                ptr_im.store(k + stride, t_im)

        current_n = n // 2

    # Radix-4 Loop
    while current_n >= 4:
        var stride = current_n // 4
        radix4_dif_simd(state, factors_re, factors_im, stride)
        current_n = current_n // 4

    # Bit Reverse
    bit_reverse_state(state)

    # Normalization
    var scale = FloatType(1.0) / sqrt(FloatType(n))
    var final_ptr_re = state.re.unsafe_ptr()
    var final_ptr_im = state.im.unsafe_ptr()

    for i in range(n):
        final_ptr_re[i] *= scale
        final_ptr_im[i] *= scale
