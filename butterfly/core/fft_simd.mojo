"""
Parallelized Fast Fourier Transform (FFT) implementation for QuantumState.

This module provides FFT operations with SIMD normalization and optional
parallelization for improved performance on large quantum states.
"""

from math import cos, sin, pi, log2, sqrt
from butterfly.core.types import *
from butterfly.core.state import QuantumState
from sys.info import simd_width_of
from algorithm import vectorize
from buffer import NDBuffer

alias simd_width = simd_width_of[Type]()


fn fft_simd[N: Int](mut state: QuantumState, inverse: Bool = False):
    """Apply FFT with SIMD normalization to the quantum state.

    This is an in-place FFT using the radix-2 decimation-in-time algorithm
    with SIMD-optimized normalization for inverse transforms.

    Args:
        N: Size of the state (must match state.size())
        state: QuantumState to transform (modified in-place)
        inverse: If True, performs inverse FFT
    """
    var n = state.size()
    var log_n = Int(log2(Float64(n)))

    # Bit-reversal permutation (scalar - not easily vectorizable)
    for i in range(n):
        var j = bit_reverse_index(i, log_n)
        if j > i:
            # Swap state[i] and state[j]
            var temp_re = state.re[i]
            var temp_im = state.im[i]
            state.re[i] = state.re[j]
            state.im[i] = state.im[j]
            state.re[j] = temp_re
            state.im[j] = temp_im

    # Create NDBuffers for SIMD operations
    var vector_re = NDBuffer[Type, 1, _, N](state.re)
    var vector_im = NDBuffer[Type, 1, _, N](state.im)

    # Cooley-Tukey FFT with SIMD butterflies
    var m = 2
    while m <= n:
        var angle = -2.0 * pi / FloatType(m)
        if inverse:
            angle = -angle

        # Process all groups at this stage
        # Use iterative twiddle factor update (faster than cos/sin per element)
        var wm_re = cos(angle)
        var wm_im = sin(angle)

        for k in range(0, n, m):
            var half_m = m // 2
            var w_re = FloatType(1.0)
            var w_im = FloatType(0.0)

            for j in range(half_m):
                var j_idx = k + j
                var u_re = state.re[j_idx]
                var u_im = state.im[j_idx]
                var v_re = state.re[j_idx + half_m]
                var v_im = state.im[j_idx + half_m]

                # Complex multiplication: t = w * v
                var t_re = w_re * v_re - w_im * v_im
                var t_im = w_re * v_im + w_im * v_re

                # Butterfly: u +/- t
                state.re[j_idx] = u_re + t_re
                state.im[j_idx] = u_im + t_im
                state.re[j_idx + half_m] = u_re - t_re
                state.im[j_idx + half_m] = u_im - t_im

                # Update twiddle factor iteratively (avoids cos/sin calls)
                var w_re_new = w_re * wm_re - w_im * wm_im
                var w_im_new = w_re * wm_im + w_im * wm_re
                w_re = w_re_new
                w_im = w_im_new

        m *= 2

    # Standard FFT normalization: divide by N for inverse
    if inverse:
        var scale = FloatType(1.0) / FloatType(n)

        @parameter
        fn normalize_simd[width: Int](i: Int):
            var re = vector_re.load[width=width](i)
            var im = vector_im.load[width=width](i)
            vector_re.store[width=width](i, re * scale)
            vector_im.store[width=width](i, im * scale)

        vectorize[normalize_simd, simd_width](n)


fn ifft_simd[N: Int](mut state: QuantumState):
    """Apply SIMD-optimized inverse FFT to the quantum state.

    Args:
        N: Size of the state (must match state.size())
        state: QuantumState to transform (modified in-place)
    """
    fft_simd[N](state, inverse=True)


fn bit_reverse_index(idx: Int, num_bits: Int) -> Int:
    """Reverse the bits of an index for FFT bit-reversal permutation.

    Args:
        idx: Index to reverse
        num_bits: Number of bits to consider

    Returns:
        Bit-reversed index
    """
    var result = 0
    var x = idx
    for _ in range(num_bits):
        result = (result << 1) | (x & 1)
        x >>= 1
    return result


fn fft_simd_parallel[N: Int](mut state: QuantumState, inverse: Bool = False):
    """Apply parallel FFT with multi-threading and SIMD normalization.

    This version uses parallelization across FFT groups for improved performance
    on large states with many groups (N >= 16). Also includes SIMD normalization.

    Args:
        N: Size of the state (must match state.size())
        state: QuantumState to transform (modified in-place)
        inverse: If True, performs inverse FFT
    """
    from algorithm import parallelize

    var n = state.size()
    var log_n = Int(log2(Float64(n)))

    # Bit-reversal permutation
    for i in range(n):
        var j = bit_reverse_index(i, log_n)
        if j > i:
            var temp_re = state.re[i]
            var temp_im = state.im[i]
            state.re[i] = state.re[j]
            state.im[i] = state.im[j]
            state.re[j] = temp_re
            state.im[j] = temp_im

    var vector_re = NDBuffer[Type, 1, _, N](state.re)
    var vector_im = NDBuffer[Type, 1, _, N](state.im)

    # Cooley-Tukey FFT
    var m = 2
    while m <= n:
        var angle = -2.0 * pi / FloatType(m)
        if inverse:
            angle = -angle

        var num_groups = n // m
        var wm_re = cos(angle)
        var wm_im = sin(angle)

        # Parallelize over groups when there are enough of them
        if num_groups >= 4:
            @parameter
            fn process_group(group_idx: Int):
                var k = group_idx * m
                var half_m = m // 2
                var w_re = FloatType(1.0)
                var w_im = FloatType(0.0)

                # Process butterflies in this group
                for j in range(half_m):
                    var j_idx = k + j
                    var u_re = state.re[j_idx]
                    var u_im = state.im[j_idx]
                    var v_re = state.re[j_idx + half_m]
                    var v_im = state.im[j_idx + half_m]

                    var t_re = w_re * v_re - w_im * v_im
                    var t_im = w_re * v_im + w_im * v_re

                    state.re[j_idx] = u_re + t_re
                    state.im[j_idx] = u_im + t_im
                    state.re[j_idx + half_m] = u_re - t_re
                    state.im[j_idx + half_m] = u_im - t_im

                    # Update twiddle factor
                    var w_re_new = w_re * wm_re - w_im * wm_im
                    var w_im_new = w_re * wm_im + w_im * wm_re
                    w_re = w_re_new
                    w_im = w_im_new

            parallelize[process_group](num_groups, num_groups)
        else:
            # Sequential for small number of groups
            for k in range(0, n, m):
                var half_m = m // 2
                var w_re = FloatType(1.0)
                var w_im = FloatType(0.0)

                for j in range(half_m):
                    var j_idx = k + j
                    var u_re = state.re[j_idx]
                    var u_im = state.im[j_idx]
                    var v_re = state.re[j_idx + half_m]
                    var v_im = state.im[j_idx + half_m]

                    var t_re = w_re * v_re - w_im * v_im
                    var t_im = w_re * v_im + w_im * v_re

                    state.re[j_idx] = u_re + t_re
                    state.im[j_idx] = u_im + t_im
                    state.re[j_idx + half_m] = u_re - t_re
                    state.im[j_idx + half_m] = u_im - t_im

                    # Update twiddle factor
                    var w_re_new = w_re * wm_re - w_im * wm_im
                    var w_im_new = w_re * wm_im + w_im * wm_re
                    w_re = w_re_new
                    w_im = w_im_new

        m *= 2

    # Normalization
    if inverse:
        var scale = FloatType(1.0) / FloatType(n)

        @parameter
        fn normalize_simd[width: Int](i: Int):
            var re = vector_re.load[width=width](i)
            var im = vector_im.load[width=width](i)
            vector_re.store[width=width](i, re * scale)
            vector_im.store[width=width](i, im * scale)

        vectorize[normalize_simd, simd_width](n)


fn power_spectrum_simd[N: Int](state: QuantumState) -> List[FloatType]:
    """Compute the power spectrum using FFT with SIMD power computation.

    Args:
        N: Size of the state (must match state.size())
        state: QuantumState to analyze

    Returns:
        List of power values (|FFT[k]|²)
    """
    var s = state
    fft_simd[N](s)

    var n = s.size()
    var power = List[FloatType](capacity=n)
    var vector_re = NDBuffer[Type, 1, _, N](s.re)
    var vector_im = NDBuffer[Type, 1, _, N](s.im)

    # Vectorized power computation
    @parameter
    fn compute_power_simd[width: Int](i: Int):
        var re = vector_re.load[width=width](i)
        var im = vector_im.load[width=width](i)
        var mag_sq = re * re + im * im

        # Store results (need to do element by element)
        for lane in range(width):
            if i + lane < n:
                power.append(mag_sq[lane])

    vectorize[compute_power_simd, simd_width](n)

    return power^
