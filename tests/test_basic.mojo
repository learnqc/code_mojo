from butterfly.core.state import *
from butterfly.utils.common import *
from butterfly.utils.state import *


def main():
    alias n = 6
    alias N = 1 << n


    f = transform_simd[N]
#     f = transform[0]

    gates = [X, Y, Z, H, P(pi/3), RX(pi/3), RY(pi/5), RZ(pi/7)]

    state = init_state(n)
    for t in range(n):
        f(state, t, H)

    iterations: UInt = 1_000
    start = time.perf_counter_ns()
    print("\n***************** {} iterations of Mojo transform for {} qubits"
        .format(iterations, n))
    for i in range(Int(iterations)):
        f(state, i%n, gates[i%len(gates)])
    elapsed = time.perf_counter_ns() - start
    print("average time for transform: {} ns\n".format(elapsed/iterations))
    for i in range(min(len(state), 64)):
        a = state[i]
        print(Amplitude(round(a.re, 3), round(a.im, 3)), end=" | ")

    st = init_state(n)
    for i in range(len(gates)):
        f(st, i%n, gates[i])
    # print('\n', len(st))
    table = to_table(st)
    print('\n')
    for i in range(len(table)):
        print('\n')
        for j in range(len(table[i])):
            print(table[i][j], end='---' if i == 0 or i == 2 or i == len(table)-1 else ' ￤')


    print('\n')
    for i in range(min(len(st), 64)):
        a = st[i]
        print(Amplitude(round(a.re, 3), round(a.im, 3)), end=" | ")

    norm: FloatType = 0.0
    for a in st:
        norm += a.re*a.re + a.im*a.im

    assert_true(is_close(Amplitude(norm, 0), Amplitude(1, 0)))
