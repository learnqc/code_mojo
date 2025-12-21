"""
FFT V4: Synthesis of PhastFT and V3.
Global: Table-based + Twiddle Packing (Phast Trick).
Local: On-the-fly Cache-blocked (V3 Trick).
"""
from math import pi, cos, sin, sqrt, log2
from memory import UnsafePointer
from algorithm import parallelize, vectorize
from sys.info import simd_width_of
from butterfly.core.types import Amplitude
from butterfly.core.state import QuantumState, bit_reverse_state
from butterfly.core.classical_fft import generate_factors
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


fn fft_v4_kernel(
    mut state: QuantumState,
    factors_re: List[Type],
    factors_im: List[Type],
    block_log: Int = 11,
):
    """Core butterfly kernel for FFT V4."""
    var n = state.size()
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
    for stage in range(log_n - 1, block_log - 1, -1):
        var num_groups = n // 2 // stride

        # Parallel Twiddle Packing
        @parameter
        fn pack_worker(i: Int):
            var idx = i * simd_width
            var src_idxs = SIMD[DType.int64, simd_width]()
            for k in range(simd_width):
                src_idxs[k] = (idx + k) * num_groups
            ptr_tw_re.store[width=simd_width](idx, ptr_fac_re.gather(src_idxs))
            ptr_tw_im.store[width=simd_width](idx, ptr_fac_im.gather(src_idxs))

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


fn fft_v4(mut state: QuantumState, block_log: Int = 11):
    var n = state.size()
    ref factors_re, factors_im = generate_factors(n)

    fft_v4_kernel(state, factors_re, factors_im, block_log)

    bit_reverse_state(state)
    var scale = Float64(1.0) / sqrt(Float64(n))
    for i in range(n):
        state.re[i] *= scale
        state.im[i] *= scale
