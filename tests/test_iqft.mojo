from butterfly.core.state import *
from butterfly.utils.visualization import print_state


def main():
    alias n: Int = 3
    re: List[FloatType] = List[FloatType](length=2**n, fill=0.0)
    im: List[FloatType] = List[FloatType](length=2**n, fill=0.0)
    re[1] = 1.0
    state = QuantumState(re^, im^)
    iqft(state, [j for j in range(n)], swap=True)
    print_state(state)
