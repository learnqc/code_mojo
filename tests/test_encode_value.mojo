from butterfly.core.state import *
from butterfly.utils.utils import *
from butterfly.algos.value_encoding import encode_value

def main():
    n: UInt = 3
    v: FloatType = 4.7

    state = init_state(n)

    iterations: UInt = 5
    start = time.perf_counter_ns()
    print("\n***************** {} iterations of Mojo encode_value({}, {})"
        .format(iterations, n, v))
    for _ in range(iterations):
        state = encode_value(n, v)
    elapsed = time.perf_counter_ns() - start
    print("average time for encode_value: {} ns\n".format(elapsed/iterations))
    print_state(state)
