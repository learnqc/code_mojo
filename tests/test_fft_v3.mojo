"""
Test FFT V3 implementation for correctness.
"""
from butterfly.core.state import QuantumState
from butterfly.core.fft_v3 import fft_v3
from butterfly.core.classical_fft import fft_dif_simd
from testing import assert_almost_equal


fn test_fft_v3() raises:
    """Test FFT v3 against reference implementation."""
    alias n = 10
    alias size = 1 << n

    # Create test state
    var state1 = QuantumState(n)
    var state2 = QuantumState(n)

    # Initialize with some values
    for i in range(size):
        var val = Float64(i) / Float64(size)
        state1.re[i] = val
        state1.im[i] = val * 0.5
        state2.re[i] = val
        state2.im[i] = val * 0.5

    # Apply FFT v3 (block_log=10 tests only Phase 2)
    fft_v3(state1, block_log=10)

    # Apply reference FFT
    fft_dif_simd(state2)

    # Compare results
    print("Testing FFT v3 correctness...")
    var max_error: Float64 = 0.0
    for i in range(size):
        var err_re = abs(state1.re[i] - state2.re[i])
        var err_im = abs(state1.im[i] - state2.im[i])
        if err_re > max_error:
            max_error = err_re
        if err_im > max_error:
            max_error = err_im

    print("Max error:", max_error)
    if max_error < 1e-10:
        print("✅ FFT v3 is CORRECT!")
    else:
        print("❌ FFT v3 has errors!")

    assert_almost_equal(max_error, 0.0, atol=1e-10)


fn main() raises:
    test_fft_v3()
