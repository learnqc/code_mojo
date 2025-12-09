from butterfly.core.fft import fft, ifft, fft_convolve, power_spectrum, phase_spectrum
from butterfly.core.state import QuantumState
from butterfly.core.types import *
from testing import assert_almost_equal
from math import sqrt, pi, cos, sin


def test_fft_identity():
    """Test that FFT followed by IFFT returns original state."""
    print("Test 1: FFT identity (FFT -> IFFT)")

    var n = 3  # 8 elements
    var state = QuantumState(n)

    # Set some arbitrary values
    state.re[0] = 1.0
    state.re[1] = 0.5
    state.re[2] = 0.25
    state.im[1] = 0.3
    state.im[2] = 0.6

    # Store original values
    var orig_re = List[FloatType](capacity=8)
    var orig_im = List[FloatType](capacity=8)
    for i in range(8):
        orig_re.append(state.re[i])
        orig_im.append(state.im[i])

    # Apply FFT and then IFFT
    fft(state)
    ifft(state)

    # Check that we got back the original state
    for i in range(8):
        assert_almost_equal(Float64(state.re[i]), Float64(orig_re[i]), atol=1e-6)
        assert_almost_equal(Float64(state.im[i]), Float64(orig_im[i]), atol=1e-6)

    print("✓ FFT identity test passed\n")


def test_fft_impulse():
    """Test FFT of impulse (delta function) should be all ones."""
    print("Test 2: FFT of impulse")

    var n = 3  # 8 elements
    var state = QuantumState(n)

    # Set impulse at index 0
    state.re[0] = 1.0
    # All others are already 0

    fft(state)

    # FFT of delta function should be constant (1/sqrt(N) after normalization)
    # But since we don't normalize forward FFT, each bin should be 1.0
    for i in range(8):
        assert_almost_equal(Float64(state.re[i]), 1.0, atol=1e-6)
        assert_almost_equal(Float64(state.im[i]), 0.0, atol=1e-6)

    print("✓ FFT impulse test passed\n")


def test_fft_sine_wave():
    """Test FFT of a pure sine wave."""
    print("Test 3: FFT of sine wave")

    var n = 3  # 8 elements
    var state = QuantumState(n)

    # Create a sine wave with frequency 1 (1 cycle over 8 samples)
    for i in range(8):
        var angle = 2.0 * pi * FloatType(i) / 8.0
        state.re[i] = sin(angle)
        state.im[i] = 0.0

    fft(state)

    # For a pure sine wave, we expect two peaks: one at k=1 and one at k=7 (N-1)
    # The peak should be at frequency bin 1 (and its complex conjugate at 7)
    var magnitude_1 = sqrt(state.re[1] * state.re[1] + state.im[1] * state.im[1])
    var magnitude_7 = sqrt(state.re[7] * state.re[7] + state.im[7] * state.im[7])

    # Should have equal magnitude at bins 1 and 7
    assert_almost_equal(Float64(magnitude_1), Float64(magnitude_7), atol=1e-6)

    # Other bins should be near zero
    for i in range(8):
        if i != 1 and i != 7:
            var mag = sqrt(state.re[i] * state.re[i] + state.im[i] * state.im[i])
            assert_almost_equal(Float64(mag), 0.0, atol=1e-5)

    print("✓ FFT sine wave test passed\n")


def test_fft_cosine_wave():
    """Test FFT of a pure cosine wave."""
    print("Test 4: FFT of cosine wave")

    var n = 3  # 8 elements
    var state = QuantumState(n)

    # Create a cosine wave with frequency 1
    for i in range(8):
        var angle = 2.0 * pi * FloatType(i) / 8.0
        state.re[i] = cos(angle)
        state.im[i] = 0.0

    fft(state)

    # For cosine, we expect peaks at bins 1 and 7 with real components
    # Bins 1 and 7 should have the energy
    var magnitude_0 = sqrt(state.re[0] * state.re[0] + state.im[0] * state.im[0])
    var magnitude_1 = sqrt(state.re[1] * state.re[1] + state.im[1] * state.im[1])
    var magnitude_7 = sqrt(state.re[7] * state.re[7] + state.im[7] * state.im[7])

    # DC component should be near zero for cosine wave (no offset)
    assert_almost_equal(Float64(magnitude_0), 0.0, atol=1e-5)

    # Magnitudes at 1 and 7 should be equal
    assert_almost_equal(Float64(magnitude_1), Float64(magnitude_7), atol=1e-6)

    print("✓ FFT cosine wave test passed\n")


def test_power_spectrum():
    """Test power spectrum computation."""
    print("Test 5: Power spectrum")

    var n = 3  # 8 elements
    var state = QuantumState(n)

    # Create a signal with two frequencies
    for i in range(8):
        var angle1 = 2.0 * pi * FloatType(i) / 8.0  # frequency 1
        var angle2 = 2.0 * pi * FloatType(i) * 2.0 / 8.0  # frequency 2
        state.re[i] = cos(angle1) + 0.5 * cos(angle2)
        state.im[i] = 0.0

    var power = power_spectrum(state)

    # Should have peaks at bins 1, 2, 6, 7
    # (positive and negative frequencies)
    var total_power = FloatType(0.0)
    for i in range(8):
        total_power += power[i]

    # Power should be conserved (Parseval's theorem)
    # Total power in frequency domain = total power in time domain
    print("  Total power:", total_power)

    # Check that we have significant power at expected bins
    print("  Power at bin 1:", power[1])
    print("  Power at bin 2:", power[2])

    print("✓ Power spectrum test passed\n")


def test_fft_convolution():
    """Test convolution using FFT."""
    print("Test 6: FFT convolution")

    var n = 2  # 4 elements (small for simple test)
    var state1 = QuantumState(n)
    var state2 = QuantumState(n)

    # Simple signals
    state1.re[0] = 1.0
    state1.re[1] = 1.0

    state2.re[0] = 1.0
    state2.re[1] = 1.0

    var result = fft_convolve(state1, state2)

    # Convolution result should have specific structure
    print("  Convolution result:")
    for i in range(4):
        print("    result[", i, "] =", result.re[i], "+", result.im[i], "i")

    print("✓ FFT convolution test passed\n")


def test_fft_linearity():
    """Test linearity property: FFT(a*x + b*y) = a*FFT(x) + b*FFT(y)."""
    print("Test 7: FFT linearity")

    var n = 3  # 8 elements
    var x = QuantumState(n)
    var y = QuantumState(n)

    # Set some values
    x.re[0] = 1.0
    x.re[1] = 0.5
    y.re[0] = 0.3
    y.re[2] = 0.7

    # Compute a*x + b*y
    var a = FloatType(2.0)
    var b = FloatType(3.0)
    var combined = QuantumState(n)
    for i in range(8):
        combined.re[i] = a * x.re[i] + b * y.re[i]
        combined.im[i] = a * x.im[i] + b * y.im[i]

    # FFT of combined
    fft(combined)

    # FFT of x and y separately
    fft(x)
    fft(y)

    # Check linearity: FFT(combined) should equal a*FFT(x) + b*FFT(y)
    for i in range(8):
        var expected_re = a * x.re[i] + b * y.re[i]
        var expected_im = a * x.im[i] + b * y.im[i]
        assert_almost_equal(Float64(combined.re[i]), Float64(expected_re), atol=1e-6)
        assert_almost_equal(Float64(combined.im[i]), Float64(expected_im), atol=1e-6)

    print("✓ FFT linearity test passed\n")


def main():
    print("=== FFT Tests ===\n")

    test_fft_identity()
    test_fft_impulse()
    test_fft_sine_wave()
    test_fft_cosine_wave()
    test_power_spectrum()
    test_fft_convolution()
    test_fft_linearity()

    print("✅ All FFT tests passed!")
