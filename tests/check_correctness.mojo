from butterfly.core.state import *
from butterfly.core.gates import *
from testing import assert_almost_equal


fn main() raises:
    alias n = 4
    alias t = 0

    print("Checking correctness of X gate twice on target", t)

    var state = init_state(n)
    # Initial state is |0...0> = [1, 0, ...]

    print("Initial state:")
    # print(state.re[0], state.im[0])

    # Apply X once
    transform_simd[1 << n](state, t, X)
    print("After 1st X:")
    # Should be |...1> (if t=0 is LSB) or |1...> (if MSB).
    # transform_simd uses stride = 1 << target.
    # If target=0, stride=1.
    # Pairs are (0, 1), (2, 3)...
    # X swaps them.
    # So index 0 should be 0, index 1 should be 1.
    print("re[0]:", state.re[0], "re[1]:", state.re[1])

    if state.re[0] != 0.0 or state.re[1] != 1.0:
        print("ERROR: 1st X gate failed!")

    # Apply X again
    transform_simd[1 << n](state, t, X)
    print("After 2nd X:")
    print("re[0]:", state.re[0], "re[1]:", state.re[1])

    if state.re[0] != 1.0 or state.re[1] != 0.0:
        print("ERROR: 2nd X gate failed! State should be restored.")
    else:
        print("SUCCESS: State restored.")
