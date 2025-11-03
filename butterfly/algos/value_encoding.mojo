from butterfly.core.state import *

def encode_value(n: UInt, v: FloatType) -> State:
    state = init_state(n)

    for j in range(n):
        transform(state, j, H)

    for j in range(n):

        # transform(state, j, P(2 * pi / 2 ** (n - j) * v))
         transform(state, j, P(2 * pi / 2 ** (j + 1) * v))

    iqft(state, [n-1-j for j in range(n)])
    return state^