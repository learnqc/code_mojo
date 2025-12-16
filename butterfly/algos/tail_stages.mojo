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
        var base = idx * 8

        # --- Stage 2 (Stride 2 -> Dist 4) ---
        alias width = 4
        var v0_re = ptr_re.load[width=width](base)
        var v0_im = ptr_im.load[width=width](base)

        var v1_re = ptr_re.load[width=width](base + 4)
        var v1_im = ptr_im.load[width=width](base + 4)

        # Butterfly Stride 2
        var sum_re = v0_re + v1_re
        var sum_im = v0_im + v1_im
        var diff_re = v0_re - v1_re
        var diff_im = v0_im - v1_im

        # Apply Twiddle for Stride 2: 1, -i, 1, -i
        # Mask for odd positions (1, 3): (0, 1, 0, 1)
        # -i rotation: (re, im) -> (im, -re)

        var rot_re = diff_im
        var rot_im = -diff_re

        var mask = SIMD[DType.bool, width](0, 1, 0, 1)

        var v1_final_re = mask.select(rot_re, diff_re)
        var v1_final_im = mask.select(rot_im, diff_im)

        # --- Stage 1 (Stride 1 -> Dist 1) ---
        # Chunk 0 (Processing 'sum' from Stage 2)
        var parts_re = sum_re.deinterleave()
        var ve0_re = parts_re[0]
        var vo0_re = parts_re[1]
        var s0_re = ve0_re + vo0_re
        var d0_re = ve0_re - vo0_re
        var mixed0_re = s0_re.interleave(d0_re)

        var parts_im = sum_im.deinterleave()
        var ve0_im = parts_im[0]
        var vo0_im = parts_im[1]
        var s0_im = ve0_im + vo0_im
        var d0_im = ve0_im - vo0_im
        var mixed0_im = s0_im.interleave(d0_im)

        ptr_re.store(base, mixed0_re)
        ptr_im.store(base, mixed0_im)

        # Chunk 1 (Processing 'v1_final' from Stage 2)
        var parts1_re = v1_final_re.deinterleave()
        var ve1_re = parts1_re[0]
        var vo1_re = parts1_re[1]
        var s1_re = ve1_re + vo1_re
        var d1_re = ve1_re - vo1_re
        var mixed1_re = s1_re.interleave(d1_re)

        var parts1_im = v1_final_im.deinterleave()
        var ve1_im = parts1_im[0]
        var vo1_im = parts1_im[1]
        var s1_im = ve1_im + vo1_im
        var d1_im = ve1_im - vo1_im
        var mixed1_im = s1_im.interleave(d1_im)

        ptr_re.store(base + 4, mixed1_re)
        ptr_im.store(base + 4, mixed1_im)

    parallelize[worker](n // 8)


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
        var base = idx * 8

        # Load indices 0..3 and 4..7
        # Bit 2 (4) clear: 0..3. Bit 2 set: 4..7.
        alias width = 4
        var top_re = ptr_re.load[width=width](base)
        var top_im = ptr_im.load[width=width](base)
        var bot_re = ptr_re.load[width=width](base + 4)
        var bot_im = ptr_im.load[width=width](base + 4)

        # Gather Twiddles
        # Indices 0..3. Logical j = indices % 4 (which is indices).
        var idxs = SIMD[DType.int64, width](0, 1, 2, 3)
        var fac_idxs = idxs * factor_stride

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
        ptr_re.store(base + 4, t_re)
        ptr_im.store(base + 4, t_im)

    parallelize[worker](n // 8)


fn butterfly_dist4_swapped_generic(
    mut state: QuantumState,
    ptr_fac_re: UnsafePointer[FloatType],
    ptr_fac_im: UnsafePointer[FloatType],
    target: Int,
    factor_stride: Int,
):
    """
    Experimental kernel: Performs butterfly for stride = 1<<target,
    assuming bit 'target' has been swapped with bit 2 (Distance 4).
    """
    var n = state.size()
    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()

    var stride = 1 << target

    # Masks for reconstructing j
    var mask_low = 3  # Bits 0, 1
    # Bits 3..target-1
    # If target <= 2, this is 0.
    var mask_mid = (stride - 1) & ~7

    @parameter
    fn worker(idx: Int):
        var base = idx * 8
        alias width = 4

        var top_re = ptr_re.load[width=width](base)
        var top_im = ptr_im.load[width=width](base)
        var bot_re = ptr_re.load[width=width](base + 4)
        var bot_im = ptr_im.load[width=width](base + 4)

        # Calculate j indices for twiddles
        # vector idxs: [base, base+1, base+2, base+3]
        var idxs = SIMD[DType.int64, width](0, 1, 2, 3) + base

        var j_reconstructed: SIMD[DType.int64, width]

        if target < 2:
            j_reconstructed = idxs & (stride - 1)
        else:
            var j_low = idxs & mask_low
            var mid = idxs & mask_mid
            var bit_from_target = (idxs >> target) & 1
            j_reconstructed = j_low | (bit_from_target << 2) | mid

        var fac_idxs = j_reconstructed * factor_stride

        var w_re = ptr_fac_re.gather(fac_idxs)
        var w_im = ptr_fac_im.gather(fac_idxs)

        var diff_re = top_re - bot_re
        var diff_im = top_im - bot_im
        var sum_re = top_re + bot_re
        var sum_im = top_im + bot_im

        var t_re = diff_re * w_re - diff_im * w_im
        var t_im = diff_re * w_im + diff_im * w_re

        ptr_re.store(base, sum_re)
        ptr_im.store(base, sum_im)
        ptr_re.store(base + 4, t_re)
        ptr_im.store(base + 4, t_im)

    parallelize[worker](n // 8)
