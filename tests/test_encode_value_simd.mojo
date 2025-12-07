from butterfly.core.state import *
from butterfly.utils.visualization import print_state
from butterfly.algos.value_encoding import (
    encode_value_interval,
    encode_value_simd,
    encode_value_simd_interval,
)


def main():
    alias n: Int = 15  # non-SIMD interval is better for n <= 14
    v: FloatType = 4.7

    state = init_state(n)

    iterations: Int = 10000
    start = time.perf_counter_ns()
    print(
        "\n***************** {} iterations of Mojo"
        " encode_value_interval({}, {})".format(iterations, n, v)
    )
    for _ in range(iterations):
        state = encode_value_interval(n, v)
    elapsed = time.perf_counter_ns() - start
    print(
        "average time for encode_value: {} ms\n".format(
            elapsed / UInt(iterations) / 1000
        )
    )
    print_state(state)

    iterations: Int = 5
    start = time.perf_counter_ns()
    print(
        "\n***************** {} iterations of Mojo encode_value_simd({}, {})"
        .format(iterations, n, v)
    )
    for _ in range(iterations):
        state = encode_value_simd[n](v)
    elapsed = time.perf_counter_ns() - start
    print(
        "average time for encode_value: {} ms\n".format(
            elapsed / UInt(iterations) / 1000
        )
    )
    print_state(state)

    start = time.perf_counter_ns()
    print(
        "\n***************** {} iterations of Mojo"
        " encode_value_simd_interval({}, {})".format(iterations, n, v)
    )
    for _ in range(iterations):
        state = encode_value_simd_interval[n](v)
    elapsed = time.perf_counter_ns() - start
    print(
        "average time for encode_value: {} ms\n".format(
            elapsed / UInt(iterations) / 1000
        )
    )
    print_state(state)
