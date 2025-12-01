from butterfly.core.state import *
from butterfly.utils.visualization import print_state
from butterfly.algos.value_encoding import encode_value


def main():
    n: Int = 20
    v: FloatType = 4.7

    state = init_state(n)

    iterations: Int = 5
    start = time.perf_counter_ns()
    print(
        "\n***************** {} iterations of Mojo encode_value({}, {})".format(
            iterations, n, v
        )
    )
    for _ in range(iterations):
        state = encode_value(n, v)
    elapsed = time.perf_counter_ns() - start
    print(
        "average time for encode_value: {} ns\n".format(
            elapsed / UInt(iterations)
        )
    )
    print_state(state)
