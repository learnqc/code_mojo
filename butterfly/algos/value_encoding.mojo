from butterfly.core.state import *


def encode_value(n: Int, v: FloatType) -> State:
    state = init_state(n)

    for j in range(n):
        transform(state, j, H)

    for j in range(n):
        # transform(state, j, P(2 * pi / 2 ** (n - j) * v))
        transform(state, j, P(2 * pi / 2 ** (j + 1) * v))

    iqft(state, [n - 1 - j for j in range(n)])
    return state^


def encode_value_interval(n: Int, v: FloatType) -> State:
    state = init_state(n)

    for j in range(n):
        transform(state, j, H)

    for j in range(n):
        # transform(state, j, P(2 * pi / 2 ** (n - j) * v))
        transform(state, j, P(2 * pi / 2 ** (j + 1) * v))

    iqft_interval(state, [n - 1 - j for j in range(n)])
    return state^


def encode_value_simd[n: Int](v: FloatType) -> State:
    state = init_state(n)

    for j in range(n):
        transform_simd[1 << n](state, j, H)

    for j in range(n):
        # transform(state, j, P(2 * pi / 2 ** (n - j) * v))
        transform_simd[1 << n](state, j, P(2 * pi / 2 ** (j + 1) * v))

    iqft_simd[1 << n](state, [n - 1 - j for j in range(n)])
    return state^


def encode_value_simd_interval[n: Int](v: FloatType) -> State:
    state = init_state(n)

    for j in range(n):
        transform_simd[1 << n](state, j, H)

    for j in range(n):
        # transform(state, j, P(2 * pi / 2 ** (n - j) * v))
        transform_simd[1 << n](state, j, P(2 * pi / 2 ** (j + 1) * v))

    iqft_simd_interval[1 << n](state, [n - 1 - j for j in range(n)])
    return state^


def encode_value_swap(n: Int, v: FloatType) -> State:
    state = init_state(n)

    for j in range(n):
        transform(state, j, H)

    for j in range(n):
        # transform(state, j, P(2 * pi / 2 ** (n - j) * v))
        transform_swap(state, j, P(2 * pi / 2 ** (j + 1) * v))

    iqft(state, [n - 1 - j for j in range(n)])
    return state^


def encode_value_mix(n: Int, v: FloatType) -> State:
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
