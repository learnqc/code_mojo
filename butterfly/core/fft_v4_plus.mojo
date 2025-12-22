"""
FFT V4 Plus: Enhanced version of V4 with vectorized/parallelized scaling.

Improvements over V4:
1. Vectorized/parallelized scaling (eliminates sequential bottleneck)

This is a conservative enhancement that keeps V4's proven architecture
while optimizing the final scaling step.

Global: Table-based + Twiddle Packing (Phast Trick).
Local: On-the-fly Cache-blocked (V3 Trick).
"""
from math import pi, cos, sin, sqrt, log2
from memory import UnsafePointer
from algorithm import parallelize, vectorize
from sys.info import simd_width_of
from butterfly.core.types import Amplitude
from butterfly.core.state import QuantumState, bit_reverse_state
from butterfly.core.classical_fft import generate_factors, load_twiddle_4x
from utils.fast_div import FastDiv

alias FloatType = DType.float64
alias Type = Float64
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


fn fft_v4_plus_kernel(
    mut state: QuantumState,
    factors_re: List[Type],
    factors_im: List[Type],
    block_log: Int = 11,
):
    """Core butterfly kernel for FFT V4 Plus (identical to V4)."""
    var n = state.size()
    var n_quarter = n // 4
    var log_n = Int(log2(Float64(n)))

    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()
    var ptr_fac_re = factors_re.unsafe_ptr()
    var ptr_fac_im = factors_im.unsafe_ptr()

    # Twiddle Packing Buffer (Phast style)
    var tw_packed_re = List[Type](length=n // 2, fill=0.0)
    var tw_packed_im = List[Type](length=n // 2, fill=0.0)
    var ptr_tw_re = tw_packed_re.unsafe_ptr()
    var ptr_tw_im = tw_packed_im.unsafe_ptr()

    # Phase 1: Global Stages
    var stride = n // 2
    for _ in range(log_n - 1, block_log - 1, -1):
        var num_groups = n // 2 // stride

        # Parallel Twiddle Packing
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

            # LOAD PACKED TWIDDLES (Contiguous!)
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

    # Phase 2: Local Stages (V3 Style - On-the-fly)
    var effective_block_log = min(block_log, log_n)
    if effective_block_log > 0:
        var local_n = 1 << effective_block_log
        var num_local_blocks = n >> effective_block_log

        @parameter
        fn local_block_worker(blk_idx: Int):
            var blk_start = blk_idx << effective_block_log
            for stage in range(effective_block_log - 1, -1, -1):
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

        parallelize[local_block_worker](num_local_blocks)


fn fft_v4_plus(mut state: QuantumState, block_log: Int = 11, adaptive: Bool = True):
    """
    FFT V4 Plus: Enhanced FFT with vectorized/parallelized scaling.

    Args:
        state: The quantum state to transform
        block_log: Cache blocking parameter (default: 11 for 2KB blocks)
        adaptive: Use adaptive scaling strategy (default: True)
                  - For very small N (< 2^10), uses sequential (avoid overhead)
                  - For medium-large N (2^10 to 2^28), uses parallel+vectorized
                  - For very large N (> 2^28), uses sequential (avoid memory instability)
    """
    var n = state.size()
    ref factors_re, factors_im = generate_factors(n)

    fft_v4_plus_kernel(state, factors_re, factors_im, block_log)

    bit_reverse_state(state)
    # *** IMPROVEMENT: Vectorized/Parallelized Scaling (NEW IN V4 PLUS!) ***
    var scale = Float64(1.0) / sqrt(Float64(n))

    # Adaptive scaling strategy with both lower and upper bounds
    # Lower: 2^10 = 1024 elements (avoid parallelization overhead for small N)
    # Upper: 2^28 = 268M elements (avoid memory instability for very large N)
    alias PARALLEL_THRESHOLD = 1024
    alias PARALLEL_UPPER_BOUND = 268435456  # 2^28

    # Use sequential for very small (<2^10) or very large (>2^28) problems
    if adaptive and (n < PARALLEL_THRESHOLD or n > PARALLEL_UPPER_BOUND):
        # Sequential scaling (less overhead for small N, more stable for huge N)
        for i in range(n):
            state.re[i] *= scale
            state.im[i] *= scale
    else:
        # Parallel + vectorized scaling (optimal for medium-large N)
        var ptr_re = state.re.unsafe_ptr()
        var ptr_im = state.im.unsafe_ptr()

        @parameter
        fn scale_worker(vec_idx: Int):
            var idx = vec_idx * simd_width
            var re = ptr_re.load[width=simd_width](idx)
            var im = ptr_im.load[width=simd_width](idx)
            ptr_re.store[width=simd_width](idx, re * scale)
            ptr_im.store[width=simd_width](idx, im * scale)

        var vec_count = n // simd_width
        parallelize[scale_worker](vec_count)

        # Handle remainder if n is not a multiple of simd_width
        var remainder_start = vec_count * simd_width
        for i in range(remainder_start, n):
            state.re[i] *= scale
            state.im[i] *= scale
