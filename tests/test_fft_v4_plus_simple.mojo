"""
Simple test: V4 Plus should give identical results to V4.
"""
from math import sqrt
from butterfly.core.state import QuantumState, generate_state
from butterfly.core.fft_v4 import fft_v4
from butterfly.core.fft_v4_plus import fft_v4_plus


fn test_v4_plus_vs_v4() raises:
    print("Testing V4 Plus vs V4...")
    alias n = 15
    var state1 = generate_state(n)
    var state2 = state1.copy()

    fft_v4_plus(state1)
    fft_v4(state2)

    var max_diff: Float64 = 0.0
    for i in range(state1.size()):
        var diff_re = abs(state1.re[i] - state2.re[i])
        var diff_im = abs(state1.im[i] - state2.im[i])
        if diff_re > max_diff:
            max_diff = diff_re
        if diff_im > max_diff:
            max_diff = diff_im

    print("  Max difference:", max_diff)
    if max_diff < 1e-12:
        print("  ✓ V4 Plus matches V4!")
    else:
        print("  ✗ FAILED! Max diff:", max_diff)
        raise Error("V4 Plus doesn't match V4")


fn main() raises:
    test_v4_plus_vs_v4()
    print("\n✓ All tests PASSED!")
