from butterfly.core.state import *
from butterfly.core.fft import fft
from butterfly.core.fft_fma_optimized import fft_fma_opt


fn encode_value(n: Int, v: FloatType) -> State:
    state = init_state(n)

    for j in range(n):
        transform(state, j, H)

    for j in range(n):
        # transform(state, j, P(2 * pi / 2 ** (n - j) * v))
        transform(state, j, P(2 * pi / 2 ** (j + 1) * v))

    iqft(state, [n - 1 - j for j in range(n)])
    return state^


fn qfft(mut state: QuantumState, inverse: Bool = False):
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

                # Update twiddle factor
                var w_re_new = w_re * wm_re - w_im * wm_im
                var w_im_new = w_re * wm_im + w_im * wm_re
                w_re = w_re_new
                w_im = w_im_new

        m *= 2

    var scale = FloatType(1.0) / sqrt(FloatType(n))
    for i in range(n):
        state.re[i] *= scale
        state.im[i] *= scale


fn iqft_via_fft(mut state: State, inverse: Bool = False):
    fft(state, inverse=inverse)
    var norm: FloatType = 0.0
    for k in range(state.size()):
        norm += state[k].re * state[k].re + state[k].im * state[k].im
    factor = sqrt(FloatType(state.size()))
    for i in range(state.size()):
        state[i] = Amplitude(
            state[i].re / factor,
            state[i].im / factor,
        )


fn encode_value_interval[n: Int](v: FloatType) -> State:
    state = init_state(n)

    for j in range(n):
        # transform(state, j, H)
        # transform_h(state, j)
        transform_h_block_style(state, j)

    for j in range(n):
        transform(state, j, P(2 * pi / 2 ** (n - j) * v))
        # transform(state, j, P(2 * pi / 2 ** (j + 1) * v))

    # iqft_interval(state, [j for j in range(n)], swap=True)
    qfft(state)
    # fft_fma_opt[1 << n](state)
    # iqft_via_fft(state, inverse=False)

    # iqft_interval(state, [n - 1 - j for j in range(n)])

    return state^


fn encode_value_simd[n: Int](v: FloatType) -> State:
    state = init_state(n)

    for j in range(n):
        transform_simd[1 << n](state, j, H)

    for j in range(n):
        # transform(state, j, P(2 * pi / 2 ** (n - j) * v))
        transform_simd[1 << n](state, j, P(2 * pi / 2 ** (j + 1) * v))

    iqft_simd[1 << n](state, [n - 1 - j for j in range(n)])
    return state^


fn encode_value_simd_interval[n: Int](v: FloatType) -> State:
    state = init_state(n)

    for j in range(n):
        transform_simd[1 << n](state, j, H)

    for j in range(n):
        # transform(state, j, P(2 * pi / 2 ** (n - j) * v))
        transform_simd[1 << n](state, j, P(2 * pi / 2 ** (j + 1) * v))

    iqft_simd_interval[1 << n](state, [n - 1 - j for j in range(n)])
    return state^


fn encode_value_swap(n: Int, v: FloatType) -> State:
    state = init_state(n)

    for j in range(n):
        transform(state, j, H)

    for j in range(n):
        # transform(state, j, P(2 * pi / 2 ** (n - j) * v))
        transform_swap(state, j, P(2 * pi / 2 ** (j + 1) * v))

    iqft(state, [n - 1 - j for j in range(n)])
    return state^


fn encode_value_mix(n: Int, v: FloatType) -> State:
    state = init_state(n)

    threshold = 3 * n // 4

    for j in range(n):
        if j <= threshold:
            transform(state, j, H)
        else:
            transform_swap(state, j, H)

    for j in range(n):
        # transform(state, j, P(2 * pi / 2 ** (n - j) * v))

        if j <= threshold:
            transform(state, j, P(2 * pi / 2 ** (j + 1) * v))
        else:
            transform_swap(state, j, P(2 * pi / 2 ** (j + 1) * v))

    iqft(state, [n - 1 - j for j in range(n)])
    return state^
