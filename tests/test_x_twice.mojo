from butterfly.core.state import init_state, transform_simd
from butterfly.core.gates import X
from butterfly.utils.visualization import print_state
from testing import assert_true, assert_almost_equal


fn main() raises:
    print("Testing transform_simd with X gate applied twice on target 0")

    alias n = 10
    var state = init_state(n)

    print("Initial state:")
    print_state(state)

    print("\nApplying X on target 0...")
    transform_simd[1 << n](state, 0, X)
    print_state(state)

    print("\nApplying X on target 0 again...")
    transform_simd[1 << n](state, 0, X)
    print_state(state)

    # Verification
    # Expected state is |000> which has amplitude 1.0 at index 0 and 0.0 elsewhere
    print("\nVerifying state...")
    var correct = True
    if abs(state[0].re - 1.0) > 1e-5:
        correct = False
        print("Error: state[0].re should be 1.0, got", state[0].re)

    for i in range(1, 1 << n):
        if abs(state[i].re) > 1e-5 or abs(state[i].im) > 1e-5:
            correct = False
            print("Error: state[", i, "] should be 0, got", state[i])

    if correct:
        print("\nSUCCESS: State returned to original.")
    else:
        print("\nFAILURE: State did not return to original.")
