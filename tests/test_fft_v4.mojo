"""
Verification test for FFT V4.
"""
from testing import assert_almost_equal
from butterfly.core.state import QuantumState, generate_state
from butterfly.core.fft_v4 import fft_v4
from butterfly.core.classical_fft import fft_dif_simd


fn test_fft_v4() raises:
    print("Testing FFT V4 correctness...")
    alias n = 15
    var state1 = generate_state(n)
    var state2 = state1.copy()

    # Apply V4
    fft_v4(state1, block_log=10)

    # Apply reference
    fft_dif_simd(state2)

    var max_error: Float64 = 0.0
    for i in range(1 << n):
        var err_re = abs(state1.re[i] - state2.re[i])
        var err_im = abs(state1.im[i] - state2.im[i])
        max_error = max(max_error, Float64(err_re))
        max_error = max(max_error, Float64(err_im))

    print("Max error:", max_error)
    assert_almost_equal(max_error, 0.0, atol=1e-12)
    print("✅ FFT V4 is CORRECT!")


fn main() raises:
    test_fft_v4()
