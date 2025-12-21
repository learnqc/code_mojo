from butterfly.core.state import QuantumState
from butterfly.core.types import FloatType, Type, Amplitude, simd_width
from butterfly.core.gates import cis
from memory import UnsafePointer


fn phase_transform(mut state: QuantumState, theta: FloatType):
    """
    Applies the global phase gate P(theta) to all qubits.
    D|k> = e^{i * theta * popcount(k)} |k>
    """
    var n = state.size()
    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()

    var One = Amplitude(1.0, 0.0)
    var Alpha = cis(theta)

    # Recurse down to size=4 (Base case)
    recursive_phase_impl[4](ptr_re, ptr_im, One, Alpha, 0, n)


fn recursive_phase_impl[
    threshold: Int
](
    ptr_re: UnsafePointer[FloatType],
    ptr_im: UnsafePointer[FloatType],
    scale: Amplitude,
    alpha: Amplitude,
    start: Int,
    size: Int,
):
    if size <= threshold:
        phase_base_case(ptr_re, ptr_im, scale, alpha, start, size)
        return

    var stride = size >> 2
    var alpha2 = alpha * alpha

    recursive_phase_impl[threshold](ptr_re, ptr_im, scale, alpha, start, stride)
    recursive_phase_impl[threshold](
        ptr_re, ptr_im, scale * alpha, alpha, start + stride, stride
    )
    recursive_phase_impl[threshold](
        ptr_re, ptr_im, scale * alpha, alpha, start + 2 * stride, stride
    )
    recursive_phase_impl[threshold](
        ptr_re, ptr_im, scale * alpha2, alpha, start + 3 * stride, stride
    )


@always_inline
fn phase_base_case(
    ptr_re: UnsafePointer[FloatType],
    ptr_im: UnsafePointer[FloatType],
    base_scale: Amplitude,
    alpha: Amplitude,
    start: Int,
    size: Int,
):
    # Handle small sizes explicitly
    # We DO NOT launder here because we pass down to apply_scale which launders

    if size == 4:
        var s0 = base_scale
        var s1 = base_scale * alpha
        var s2 = s1
        var s3 = s1 * alpha

        apply_scale(ptr_re, ptr_im, start, s0)
        apply_scale(ptr_re, ptr_im, start + 1, s1)
        apply_scale(ptr_re, ptr_im, start + 2, s2)
        apply_scale(ptr_re, ptr_im, start + 3, s3)
    elif size == 2:
        apply_scale(ptr_re, ptr_im, start, base_scale)
        apply_scale(ptr_re, ptr_im, start + 1, base_scale * alpha)
    elif size == 1:
        apply_scale(ptr_re, ptr_im, start, base_scale)
    else:
        pass


@always_inline
fn apply_scale(
    p_re_in: UnsafePointer[FloatType],
    p_im_in: UnsafePointer[FloatType],
    idx: Int,
    s: Amplitude,
):
    # Launder pointers to ensure mutability
    # This is required because arguments are immutable references
    var p_re = UnsafePointer[FloatType](p_re_in.address)
    var p_im = UnsafePointer[FloatType](p_im_in.address)

    var r = p_re[idx]
    var i = p_im[idx]
    var new_r = r * s.re - i * s.im
    var new_i = r * s.im + i * s.re
    p_re[idx] = new_r
    p_im[idx] = new_i
