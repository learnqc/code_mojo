"""
Test FFT V4 Plus correctness against reference implementation.
"""
from math import sqrt
from testing import assert_almost_equal
from butterfly.core.state import QuantumState, generate_state
from butterfly.core.fft_v4 import fft_v4
from butterfly.core.fft_v4_plus import fft_v4_plus
from butterfly.core.classical_fft import fft_dif_simd


fn test_fft_v4_works() raises:
    print("Testing that FFT V4 (original) works correctly...")
    alias n = 15
    var state1 = generate_state(n)
    var state2 = state1.copy()

    # Apply V4
    fft_v4(state1)

    # Apply reference
    fft_dif_simd(state2)

    # Compare results
    var max_diff: Float64 = 0.0
    for i in range(state1.size()):
        var diff_re = abs(
            state1.re[i].cast[DType.float64]()
            - state2.re[i].cast[DType.float64]()
        )
        var diff_im = abs(
            state1.im[i].cast[DType.float64]()
            - state2.im[i].cast[DType.float64]()
        )
        if diff_re > max_diff:
            max_diff = diff_re
        if diff_im > max_diff:
            max_diff = diff_im

    print("  Max difference:", max_diff)
    if max_diff < 1e-10:
        print("  ✓ FFT V4 PASSED!")
    else:
        print("  ✗ FFT V4 FAILED! Max diff:", max_diff)
        raise Error("FFT V4 correctness test failed")


fn test_fft_v4_plus() raises:
    print("Testing FFT V4 Plus correctness...")
    alias n = 15
    var state1 = generate_state(n)
    var state2 = state1.copy()

    # Apply V4 Plus
    fft_v4_plus(state1)

    # Apply reference (using V4 since they should be identical except scaling)
    fft_v4(state2)

    # Compare results
    var max_diff: Float64 = 0.0
    for i in range(state1.size()):
        var diff_re = abs(
            state1.re[i].cast[DType.float64]()
            - state2.re[i].cast[DType.float64]()
        )
        var diff_im = abs(
            state1.im[i].cast[DType.float64]()
            - state2.im[i].cast[DType.float64]()
        )
        if diff_re > max_diff:
            max_diff = diff_re
        if diff_im > max_diff:
            max_diff = diff_im

    print("  Max difference:", max_diff)
    if max_diff < 1e-10:
        print("  ✓ FFT V4 Plus PASSED!")
    else:
        print("  ✗ FFT V4 Plus FAILED! Max diff:", max_diff)
        raise Error("FFT V4 Plus correctness test failed")


fn test_small_strides() raises:
    print("\nTesting specialized small-stride kernels...")

    # Test with small N to exercise stride 1, 2, 4 paths
    for n in range(4, 8):
        print("  Testing N =", n, "...")
        var state1 = generate_state(n)
        var state2 = state1.copy()

        fft_v4_plus(state1)
        fft_v4(state2)

        var max_diff: Float64 = 0.0
        for i in range(state1.size()):
            var diff_re = abs(
                state1.re[i].cast[DType.float64]()
                - state2.re[i].cast[DType.float64]()
            )
            var diff_im = abs(
                state1.im[i].cast[DType.float64]()
                - state2.im[i].cast[DType.float64]()
            )
            if diff_re > max_diff:
                max_diff = diff_re
            if diff_im > max_diff:
                max_diff = diff_im

        if max_diff > 1e-10:
            print("    ✗ FAILED at N =", n, "! Max diff:", max_diff)
            raise Error("Small stride test failed")

    print("  ✓ All small-stride tests PASSED!")


fn test_large_scale() raises:
    print("\nTesting large scale (N=20)...")
    alias n = 20
    var state1 = generate_state(n)
    var state2 = state1.copy()

    fft_v4_plus(state1)
    fft_v4(state2)

    var max_diff: Float64 = 0.0
    var sum_sq_diff: Float64 = 0.0
    for i in range(state1.size()):
        var diff_re = state1.re[i].cast[DType.float64]() - state2.re[i].cast[DType.float64]()
        var diff_im = state1.im[i].cast[DType.float64]() - state2.im[i].cast[DType.float64]()
        var diff = diff_re * diff_re + diff_im * diff_im
        sum_sq_diff += diff
        if diff > max_diff * max_diff:
            max_diff = sqrt(diff)

    print("  Max difference:", max_diff)
    print("  RMS difference:", sqrt(sum_sq_diff / Float64(state1.size())))

    if max_diff < 1e-9:
        print("  ✓ Large scale test PASSED!")
    else:
        print("  ✗ Large scale test FAILED!")
        raise Error("Large scale correctness test failed")


fn main() raises:
    print("=" * 70)
    print("FFT V4 Plus Correctness Tests")
    print("=" * 70)

    test_fft_v4_works()
    test_fft_v4_plus()
    test_small_strides()
    test_large_scale()

    print("\n" + "=" * 70)
    print("All tests PASSED! ✓")
    print("=" * 70)
