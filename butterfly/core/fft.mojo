"""
Fast Fourier Transform (FFT) implementation for QuantumState.

This module provides FFT operations that work on the QuantumState structure,
leveraging the complex number representation (re, im) already present.

Note: This is a standard Cooley-Tukey FFT implementation. While related to the
Quantum Fourier Transform (QFT), they use different conventions:
- FFT uses standard signal processing conventions with bit-reversed ordering
- QFT applies gates in specific qubit order with different phase conventions
- Use this FFT for classical signal processing on quantum states
- Use the IQFT functions in state.mojo for quantum circuit operations
"""

from math import cos, sin, pi, log2, sqrt
from butterfly.core.types import *
from butterfly.core.state import QuantumState


fn fft(mut state: QuantumState, inverse: Bool = False):
    """Apply FFT to the quantum state using Cooley-Tukey algorithm.

    This is an in-place FFT that operates on the state's complex amplitudes.
    Uses the radix-2 decimation-in-time algorithm.

    Args:
        state: QuantumState to transform (modified in-place)
        inverse: If True, performs inverse FFT
    """
    var n = state.size()
    var log_n = Int(log2(Float64(n)))

    # Bit-reversal permutation
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

    # Cooley-Tukey FFT
    var m = 2
    while m <= n:
        var angle = 2.0 * pi / FloatType(m)
        if not inverse:
            angle = -angle

        # Twiddle factor for this stage
        var wm_re = cos(angle)
        var wm_im = sin(angle)

        # Process all groups at this stage
        for k in range(0, n, m):
            var w_re = FloatType(1.0)
            var w_im = FloatType(0.0)

            # Butterfly operations within group
            for j in range(m // 2):
                var t_re = w_re * state.re[k + j + m // 2] - w_im * state.im[k + j + m // 2]
                var t_im = w_re * state.im[k + j + m // 2] + w_im * state.re[k + j + m // 2]

                var u_re = state.re[k + j]
                var u_im = state.im[k + j]

                state.re[k + j] = u_re + t_re
                state.im[k + j] = u_im + t_im
                state.re[k + j + m // 2] = u_re - t_re
                state.im[k + j + m // 2] = u_im - t_im

                # Update twiddle factor
                var w_re_new = w_re * wm_re - w_im * wm_im
                var w_im_new = w_re * wm_im + w_im * wm_re
                w_re = w_re_new
                w_im = w_im_new

        m *= 2

    # Standard FFT normalization: divide by N for inverse
    if inverse:
        var scale = FloatType(1.0) / FloatType(n)
        for i in range(n):
            state.re[i] *= scale
            state.im[i] *= scale


fn ifft(mut state: QuantumState):
    """Apply inverse FFT to the quantum state.

    Args:
        state: QuantumState to transform (modified in-place)
    """
    fft(state, inverse=True)


fn fft_convolve(state1: QuantumState, state2: QuantumState) -> QuantumState:
    """Compute convolution of two states using FFT.

    Convolution in time domain = multiplication in frequency domain.

    Args:
        state1: First quantum state
        state2: Second quantum state

    Returns:
        Convolved state
    """
    var n = state1.size()

    # Copy states for FFT
    var s1 = state1
    var s2 = state2

    # Transform to frequency domain
    fft(s1)
    fft(s2)

    # Multiply in frequency domain (complex multiplication)
    var result_re = List[FloatType](capacity=n)
    var result_im = List[FloatType](capacity=n)

    for i in range(n):
        var re = s1.re[i] * s2.re[i] - s1.im[i] * s2.im[i]
        var im = s1.re[i] * s2.im[i] + s1.im[i] * s2.re[i]
        result_re.append(re)
        result_im.append(im)

    var result = QuantumState(result_re^, result_im^)

    # Transform back to time domain
    ifft(result)

    return result^


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


fn fft_frequency_bin(mut state: QuantumState, bin_index: Int) -> Amplitude:
    """Get the amplitude at a specific frequency bin after FFT.

    This is useful for analyzing frequency components without computing full FFT.

    Args:
        state: QuantumState (will be modified by FFT)
        bin_index: Frequency bin index (0 to N-1)

    Returns:
        Complex amplitude at that frequency
    """
    fft(state)
    return state[bin_index]


fn power_spectrum(state: QuantumState) -> List[FloatType]:
    """Compute the power spectrum (magnitude squared) of the state.

    Args:
        state: QuantumState to analyze

    Returns:
        List of power values (|FFT[k]|²)
    """
    var s = state
    fft(s)

    var n = s.size()
    var power = List[FloatType](capacity=n)

    for i in range(n):
        var magnitude_sq = s.re[i] * s.re[i] + s.im[i] * s.im[i]
        power.append(magnitude_sq)

    return power^


fn phase_spectrum(state: QuantumState) -> List[FloatType]:
    """Compute the phase spectrum of the state.

    Args:
        state: QuantumState to analyze

    Returns:
        List of phase values (arg(FFT[k]))
    """
    from math import atan2

    var s = state
    fft(s)

    var n = s.size()
    var phases = List[FloatType](capacity=n)

    for i in range(n):
        var phase = atan2(s.im[i], s.re[i])
        phases.append(phase)

    return phases^
