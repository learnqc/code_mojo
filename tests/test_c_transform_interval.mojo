from butterfly.core.state import (
    QuantumState,
    c_transform,
    c_transform_interval,
    generate_state,
)
from butterfly.core.gates import X, H, P
from testing import assert_almost_equal
import math


fn check(n: Int, control: Int, target: Int) raises:
    var state = generate_state(n)
    var state_copy = generate_state(n)

    # Use a non-trivial gate
    var gate = H

    c_transform(state, control, target, gate)
    c_transform_interval(state_copy, control, target, gate)

    for i in range(state.size()):
        assert_almost_equal(
            state[i].re,
            state_copy[i].re,
            msg="Real part mismatch at index " + String(i),
        )
        assert_almost_equal(
            state[i].im,
            state_copy[i].im,
            msg="Imag part mismatch at index " + String(i),
        )


fn main() raises:
    print("Testing c_transform_interval...")

    # Test cases for N=4
    var n = 4

    print("Testing Target > Control")
    check(n, 0, 1)
    check(n, 0, 2)
    check(n, 0, 3)
    check(n, 1, 2)
    check(n, 1, 3)
    check(n, 2, 3)

    print("Testing Target < Control")
    check(n, 1, 0)
    check(n, 2, 0)
    check(n, 3, 0)
    check(n, 2, 1)
    check(n, 3, 1)
    check(n, 3, 2)

    print("All tests passed!")
