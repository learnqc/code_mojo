from butterfly.core.state import *
from butterfly.core.types import float_bytes
from butterfly.algos.adjacent_pairs import swap_state_to_distance
from butterfly.algos.tail_stages import (
    stride4_swapped_simd,
)
from butterfly.algos.fused_stages import fused_stride2_stride1_swapped
from butterfly.algos.radix4 import radix4_dif_simd

from memory import UnsafePointer
from algorithm import parallelize
from buffer import NDBuffer
from utils.fast_div import FastDiv
from time import perf_counter_ns
from math import log2


fn generate_factors(
    n: Int, inverse: Bool = False
) -> Tuple[List[FloatType], List[FloatType]]:
    var angle = 2.0 * pi / FloatType(n)
    if not inverse:
        angle = -angle

    var wm_re = cos(angle)
    var wm_im = sin(angle)

    alias sqh = sqrt(0.5).cast[Type]()

    factors_re = List[FloatType](length=n // 2, fill=0.0)
    factors_im = List[FloatType](length=n // 2, fill=0.0)

    # Use parallelization for large N
    # For small N, overhead might dominate, but with 2048 grain size it's fine.
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

        # Recompute starting W for this chunk to stay independent
        # angle * start
        var thread_angle = angle * FloatType(start)
        var w_re = cos(thread_angle)
        var w_im = sin(thread_angle)

        for j in range(start, end):
            ptr_re[j] = w_re
            ptr_im[j] = w_im

            var sum = (w_re + w_im) * sqh
            var diff = (w_re - w_im) * sqh

            ptr_re[j + n // 8] = sum
            ptr_im[j + n // 8] = -diff

            ptr_re[j + n // 4] = w_im
            ptr_im[j + n // 4] = -w_re

            ptr_re[j + 3 * (n // 8)] = -diff
            ptr_im[j + 3 * (n // 8)] = -sum

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


fn fft_dif_parallel_simd_phast(
    mut state: QuantumState, inverse: Bool = False, debug_time: Bool = False
):
    """Parallelized + Vectorized DIF FFT using sequential twiddle buffering (PhastFT style).
    OPTIMIZATION: Prevents strided gathers for twiddles by pre-packing the current stage's
    twiddle sequence into a contiguous buffer. Since the same twiddle sequence repeats
    for every block in the stage, this also improves cache locality for later stages.
    """
    var n = state.size()
    var log_n = Int(log2(Float64(n)))
    ref factors_re, factors_im = generate_factors(n)

    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()
    var ptr_fac_re = factors_re.unsafe_ptr()
    var ptr_fac_im = factors_im.unsafe_ptr()

    var stride = n // 2

    # Pre-allocate a scratch buffer for twiddles using List
    var tw_max = n // 2
    var tw_buf_re = List[FloatType](capacity=tw_max)
    var tw_buf_im = List[FloatType](capacity=tw_max)

    # Fast initialization (unsafe setting length)
    # Since we are writing to it before reading, we can just "set" the size if List supported it.
    # But for now, use resize (assuming it's faster than loop)
    # Or pointer arithmetic to set size? No private member access.
    # We will assume resize is decent, or just use unsafe_ptr to write without resizing (unsafe but we can just use capacity).
    # Wait, if we use capacity but size is 0, list might free/realloc?
    # List is struct.
    # Actually, we can just use the unsafe pointer from the start and IGNORE the size property
    # as long as we don't call append/methods that rely on size.
    # We allocated capacity. The memory is there.

    # Pre-allocate buffer (always allocate to be safe, or alloc inside if needed? List alloc is fast enough if reused/outside)
    # We moved alloc outside? Yes.

    var ptr_tw_buf_re = tw_buf_re.unsafe_ptr()
    var ptr_tw_buf_im = tw_buf_im.unsafe_ptr()

    var t_start = perf_counter_ns()

    for _ in range(log_n):
        var t_stage_start = perf_counter_ns()
        var fd = FastDiv[DType.uint32](stride)
        alias TargetType = Scalar[FastDiv[DType.uint32].uint_type]

        var num_groups = n // 2 // stride

        if (stride // 2) >= simd_width:
            # Tiling Optimization for Large Strides (> L2 Cache)
            # Threshold chosen based on N=22 benchmarks (Stride 1M was slow, Tiling fixed it).
            # Compute tiling thresholds based on Cache Sizes
            # L2 Cache Target: 4MB. L1 Cache Target: 64KB.
            alias target_l2_bytes = 4 * 1024 * 1024
            alias target_l1_bytes = 64 * 1024
            alias complex_size = 2 * float_bytes

            alias TILED_THRESHOLD = target_l2_bytes // complex_size
            alias TILE_SIZE = target_l1_bytes // complex_size

            # Use Zero-Copy (Direct Gather) if strides are small enough (avoid buffer overhead)
            # Factor Stride 1 (Stage 0), 2 (Stage 1), 4 (Stage 2) are good candidates.
            if num_groups <= 4 and stride >= TILED_THRESHOLD:
                # Tiled Zero-Copy Mode (Generic num_groups)
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

                # Handle remainder if any (unlikely with power of 2 sizes, but good practice)
                # Power of 2 sizes will always be divisible by 4096 (2^12) if n >= 13.

            elif num_groups == 1:
                # Standard Zero-Copy: Use factors directly (Origin: ptr_fac_re)
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

                    var w_re = ptr_fac_re.load[width=simd_width](j_base)
                    var w_im = ptr_fac_im.load[width=simd_width](j_base)

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
                # Buffer Mode (Origin: ptr_tw_buf_re)

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

        else:
            # Fallback (Small Strides or Large SIMD width)
            if stride == 4 and n >= 8:
                # Optimized Stride 4 (Vectorized Swap + Kernel)
                # target=2 (Stride 4) -> Swap Bit 2 with Bit 2 (No-op)
                # swap_state_to_distance_4_simd(state, 2)
                stride4_swapped_simd(state, ptr_fac_re, ptr_fac_im, num_groups)
                # swap_state_to_distance_4_simd(state, 2)
            elif stride == 2 and n >= 8:
                # Optimized Fused Stride 2 + 1
                swap_state_to_distance(state, 1, 4)
                fused_stride2_stride1_swapped(state)
                swap_state_to_distance(state, 1, 4)
                if debug_time:
                    var t_now = perf_counter_ns()
                    print(
                        "Stage Stride 2+1 (Fused): ",
                        (t_now - t_stage_start) / 1e6,
                        "ms",
                    )
                # Handled Stride 1 implicitly
                break
            else:
                # Fallback Scalar Loop
                for idx in range(n // 2):
                    var res = fd.__divmod__(TargetType(idx))
                    var block_idx = Int(res[0])
                    var j = Int(res[1])

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

    # Lists freed automatically

    if debug_time:
        var t_stages_end = perf_counter_ns()
        print("Total Stages: ", (t_stages_end - t_start) / 1e6, "ms")

    var t_bitrev_start = perf_counter_ns()
    bit_reverse_state(state)
    if debug_time:
        var t_bitrev_end = perf_counter_ns()
        print("Bit Reversal: ", (t_bitrev_end - t_bitrev_start) / 1e6, "ms")
    var scale = FloatType(1.0) / sqrt(FloatType(n))
    var final_ptr_re = state.re.unsafe_ptr()
    var final_ptr_im = state.im.unsafe_ptr()

    for i in range(n):
        final_ptr_re[i] *= scale
        final_ptr_im[i] *= scale


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
