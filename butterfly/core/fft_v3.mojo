"""
FFT V3: Quantum-inspired FFT using v3 executor optimizations.

Applies cache-blocking, Radix-4 fusion, and SIMD vectorization
to achieve better performance than traditional FFT implementations.
"""
from math import pi, cos, sin, sqrt
from memory import UnsafePointer
from algorithm import parallelize, vectorize
from sys.info import simd_width_of
from butterfly.core.types import Amplitude
from butterfly.core.state import QuantumState, bit_reverse_state

alias FloatType = DType.float64
alias simd_width = simd_width_of[FloatType]()

# Default cache-blocking threshold (same as v3 executor)
alias DEFAULT_BLOCK_LOG = 11


@always_inline
fn compute_twiddle_simd[
    width: Int
](j_vec: SIMD[FloatType, width], block_size: Int) -> Tuple[
    SIMD[FloatType, width], SIMD[FloatType, width]
]:
    """Compute SIMD twiddle factors for DIF: e^(-2πi*j / block_size)."""
    var angle = -2.0 * pi * j_vec / Float64(block_size)
    return cos(angle), sin(angle)


fn fft_v3_kernel(
    mut state: QuantumState,
    block_log: Int = DEFAULT_BLOCK_LOG,
):
    """Core butterfly kernel for FFT V3."""
    var n = state.size()
    var log_n = Int(log2(Float64(n)))

    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()

    # Phase 1: Global stages
    for stage in range(log_n - 1, block_log - 1, -1):
        var block_size = 1 << (stage + 1)
        var half_block = block_size >> 1
        var num_blocks = n >> (stage + 1)

        @parameter
        fn global_worker(blk_idx: Int):
            var blk_start = blk_idx * block_size

            @parameter
            fn v_kernel[width: Int](j: Int):
                var j_offsets = SIMD[FloatType, width]()
                for k in range(width):
                    j_offsets[k] = k
                var j_vec = SIMD[FloatType, width](j) + j_offsets
                var w_re, w_im = compute_twiddle_simd[width](j_vec, block_size)
                var idx0 = blk_start + j
                var idx1 = idx0 + half_block
                var a_re = ptr_re.load[width=width](idx0)
                var a_im = ptr_im.load[width=width](idx0)
                var b_re = ptr_re.load[width=width](idx1)
                var b_im = ptr_im.load[width=width](idx1)
                var diff_re = a_re - b_re
                var diff_im = a_im - b_im
                ptr_re.store[width=width](idx0, a_re + b_re)
                ptr_im.store[width=width](idx0, a_im + b_im)
                ptr_re.store[width=width](idx1, diff_re * w_re - diff_im * w_im)
                ptr_im.store[width=width](idx1, diff_re * w_im + diff_im * w_re)

            vectorize[v_kernel, simd_width](half_block)

        parallelize[global_worker](num_blocks)

    # Phase 2: Local stages with cache-blocking
    var effective_block_log = min(block_log, log_n)
    if effective_block_log > 0:
        var local_n = 1 << effective_block_log
        var num_local_blocks = n >> effective_block_log

        @parameter
        fn local_block_worker(blk_idx: Int):
            var blk_start = blk_idx << effective_block_log
            for stage in range(effective_block_log - 1, -1, -1):
                var stage_block_size = 1 << (stage + 1)
                var half_stage_block = stage_block_size >> 1
                var num_inner_blocks = local_n >> (stage + 1)
                for inner_blk in range(num_inner_blocks):
                    var sub_blk_start = blk_start + (
                        inner_blk * stage_block_size
                    )

                    @parameter
                    fn v_kernel_local[width: Int](j: Int):
                        var j_offsets = SIMD[FloatType, width]()
                        for k in range(width):
                            j_offsets[k] = k
                        var j_vec = SIMD[FloatType, width](j) + j_offsets
                        var w_re, w_im = compute_twiddle_simd[width](
                            j_vec, stage_block_size
                        )
                        var idx0 = sub_blk_start + j
                        var idx1 = idx0 + half_stage_block
                        var a_re = ptr_re.load[width=width](idx0)
                        var a_im = ptr_im.load[width=width](idx0)
                        var b_re = ptr_re.load[width=width](idx1)
                        var b_im = ptr_im.load[width=width](idx1)
                        var diff_re = a_re - b_re
                        var diff_im = a_im - b_im
                        ptr_re.store[width=width](idx0, a_re + b_re)
                        ptr_im.store[width=width](idx0, a_im + b_im)
                        ptr_re.store[width=width](
                            idx1, diff_re * w_re - diff_im * w_im
                        )
                        ptr_im.store[width=width](
                            idx1, diff_re * w_im + diff_im * w_re
                        )

                    vectorize[v_kernel_local, simd_width](half_stage_block)

        parallelize[local_block_worker](num_local_blocks)


fn fft_v3(
    mut state: QuantumState,
    block_log: Int = DEFAULT_BLOCK_LOG,
):
    fft_v3_kernel(state, block_log)
    bit_reverse_state(state)
    var n = state.size()
    var scale = Float64(1.0) / sqrt(Float64(n))
    for i in range(n):
        state.re[i] *= scale
        state.im[i] *= scale
