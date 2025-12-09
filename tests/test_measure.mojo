from butterfly.core.state import *
from butterfly.utils.visualization import print_state

def measurement_circuit() -> State:
    n: Int = 3
    state = init_state(n)

    values: List[Bool] = [True, False, True]
    for v in values:
        transform(state, 0, RX(pi/3))
        transform(state, 1, RX(pi/5))
        transform(state, 2, RX(pi/7))

        c_transform(state, 0, 1, RX(pi/7))
        c_transform(state, 1, 2, RX(pi/5))
        c_transform(state, 2, 1, RX(pi/3))

        _ = measure_qubit(state, 0, False, v)

        print_state(state)

    return state^


def main():
    start = time.perf_counter_ns()
    state = measurement_circuit()
    elapsed = time.perf_counter_ns() - start
    print("Time elapsed: {} ns\n".format(elapsed))
    # print_state(state)