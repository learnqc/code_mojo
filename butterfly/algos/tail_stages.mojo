from butterfly.core.state import QuantumState
from butterfly.core.types import *
from memory import UnsafePointer
from algorithm import parallelize
from butterfly.core.classical_fft import generate_factors
import math


fn fused_stride2_stride1_swapped(mut state: QuantumState):
    """
    Apply FFT stages for Stride 2 and Stride 1 simultaneously.
    Assumes the state has been pre-swapped such that Stride 2 pairs are at Distance 8.
    (Bit 1 swapped with Bit 3).

    Stride 2 pairs: (i, i+8). Twiddles: 1 for even i, -i for odd i.
    Stride 1 pairs: (i, i+1). Twiddles: 1.
    """
    var n = state.size()
    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()

    @parameter
    fn worker(idx: Int):
        var base = idx * 16

        # --- Stage 2 (Stride 2 -> Dist 8) ---
        alias width = 8
        var v0_re = ptr_re.load[width=width](base)
        var v0_im = ptr_im.load[width=width](base)

        var v1_re = ptr_re.load[width=width](base + 8)
        var v1_im = ptr_im.load[width=width](base + 8)

        # Butterfly Stride 2
        var sum_re = v0_re + v1_re
        var sum_im = v0_im + v1_im
        var diff_re = v0_re - v1_re
        var diff_im = v0_im - v1_im

        # Apply Twiddle for Stride 2: 1 for even, -i for odd
        # Even indices: 0, 2, 4, 6. Odd: 1, 3, 5, 7.
        # -i rotation: (re, im) -> (im, -re)

        var rot_re = diff_im
        var rot_im = -diff_re

        var mask = SIMD[DType.bool, width](0, 1, 0, 1, 0, 1, 0, 1)

        var v1_final_re = mask.select(rot_re, diff_re)
        var v1_final_im = mask.select(rot_im, diff_im)

        # --- Stage 1 (Stride 1 -> Dist 1) ---
        # Chunk 0 (Processing 'sum' from Stage 2)
        var ve0_re = sum_re.shuffle[0, 2, 4, 6, 0, 2, 4, 6]()
        var vo0_re = sum_re.shuffle[1, 3, 5, 7, 1, 3, 5, 7]()
        var s0_re = ve0_re + vo0_re
        var d0_re = ve0_re - vo0_re
        # interleave creates width 16
        var mixed0_re = s0_re.interleave(d0_re)

        var ve0_im = sum_im.shuffle[0, 2, 4, 6, 0, 2, 4, 6]()
        var vo0_im = sum_im.shuffle[1, 3, 5, 7, 1, 3, 5, 7]()
        var s0_im = ve0_im + vo0_im
        var d0_im = ve0_im - vo0_im
        var mixed0_im = s0_im.interleave(d0_im)

        @parameter
        for k in range(8):
            ptr_re.store(base + k, mixed0_re[k])
            ptr_im.store(base + k, mixed0_im[k])

        # Chunk 1 (Processing 'v1_final' from Stage 2)
        var ve1_re = v1_final_re.shuffle[0, 2, 4, 6, 0, 2, 4, 6]()
        var vo1_re = v1_final_re.shuffle[1, 3, 5, 7, 1, 3, 5, 7]()
        var s1_re = ve1_re + vo1_re
        var d1_re = ve1_re - vo1_re
        var mixed1_re = s1_re.interleave(d1_re)

        var ve1_im = v1_final_im.shuffle[0, 2, 4, 6, 0, 2, 4, 6]()
        var vo1_im = v1_final_im.shuffle[1, 3, 5, 7, 1, 3, 5, 7]()
        var s1_im = ve1_im + vo1_im
        var d1_im = ve1_im - vo1_im
        var mixed1_im = s1_im.interleave(d1_im)

        @parameter
        for k in range(8):
            ptr_re.store(base + 8 + k, mixed1_re[k])
            ptr_im.store(base + 8 + k, mixed1_im[k])

    parallelize[worker](n // 16)


fn stride4_swapped_simd(
    mut state: QuantumState,
    ptr_fac_re: UnsafePointer[FloatType],
    ptr_fac_im: UnsafePointer[FloatType],
    factor_stride: Int,
):
    """
    Apply FFT stage for Stride 4.
    Assumes state is swapped to Distance 8 (Bit 2 <-> Bit 3).
    """
    var n = state.size()
    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()

    @parameter
    fn worker(idx: Int):
        var base = idx * 16

        # Load indices 0..7 and 8..15
        # Bit 3 clear: 0..7. Bit 3 set: 8..15.
        # Swap logic aligned these.
        alias width = 8
        var top_re = ptr_re.load[width=width](base)
        var top_im = ptr_im.load[width=width](base)
        var bot_re = ptr_re.load[width=width](base + 8)
        var bot_im = ptr_im.load[width=width](base + 8)

        # Gather Twiddles
        # Indices 0..7. Logical j = indices % 4.
        # j sequence: 0, 1, 2, 3, 0, 1, 2, 3
        var idxs = SIMD[DType.int64, width](0, 1, 2, 3, 4, 5, 6, 7)
        var j_idxs = idxs % 4
        var fac_idxs = j_idxs * factor_stride

        var w_re = ptr_fac_re.gather(fac_idxs)
        var w_im = ptr_fac_im.gather(fac_idxs)

        # Butterfly
        var sum_re = top_re + bot_re
        var sum_im = top_im + bot_im
        var diff_re = top_re - bot_re
        var diff_im = top_im - bot_im

        var t_re = diff_re * w_re - diff_im * w_im
        var t_im = diff_re * w_im + diff_im * w_re

        # Store
        ptr_re.store(base, sum_re)
        ptr_im.store(base, sum_im)
        ptr_re.store(base + 8, t_re)
        ptr_im.store(base + 8, t_im)

    parallelize[worker](n // 16)
