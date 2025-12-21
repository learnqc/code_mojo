from butterfly.core.state import *
from butterfly.core.fft import fft
from butterfly.core.fft_fma_optimized import fft_fma_opt
from butterfly.core.fft_numpy_style import fft_numpy_style
from butterfly.core.classical_fft import (
    fft_dit,
    fft_dif,
    fft_dif_parallel,
    fft_dif_parallel_fastdiv,
    fft_dif_parallel_simd,
    fft_dif_parallel_simd_ndbuffer,
    fft_dif_parallel_simd_phast,
)


fn encode_value(n: Int, v: FloatType) -> State:
    state = init_state(n)

    for j in range(n):
        transform(state, j, H)

    for j in range(n):
        # transform(state, j, P(2 * pi / 2 ** (n - j) * v))
        transform(state, j, P(2 * pi / 2 ** (j + 1) * v))

    iqft(state, [n - 1 - j for j in range(n)])
    return state^


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
    # fft_dit(state)
    # fft_dif(state)
    # fft_dif_parallel(state)
    # fft_dif_parallel_fastdiv(state)
    fft_dif_parallel_simd_phast(state)
    # fft_dif_parallel_simd(state)
    # fft_dif_parallel_simd_ndbuffer(state)
    # fft_fma_opt[1 << n](state)
    # iqft_via_fft(state, inverse=False)
    # fft_numpy_style(state)

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


from butterfly.core.circuit import QuantumCircuit
from butterfly.algos.qft import iqft as iqft_circuit


fn encode_value_circuit(mut circuit: QuantumCircuit, n: Int, v: FloatType):
    """
    Adds value encoding gates to the circuit.

    Args:
        circuit: The QuantumCircuit to add gates to.
        n: Number of qubits.
        v: Value to encode.
    """
    for j in range(n):
        circuit.h(j)

    for j in range(n):
        circuit.p(j, 2 * pi / 2 ** (j + 1) * v)

    iqft_circuit(circuit, [n - 1 - j for j in range(n)], do_swap=True)
