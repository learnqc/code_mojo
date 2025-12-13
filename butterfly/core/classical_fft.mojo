from butterfly.core.state import *
from memory import UnsafePointer
from algorithm import parallelize


fn generate_factors(
    n: Int, inverse: Bool = False
) -> Tuple[List[FloatType], List[FloatType]]:
    var angle = 2.0 * pi / FloatType(n)
    if not inverse:
        angle = -angle

    # Twiddle factor for this stage
    var wm_re = cos(angle)
    var wm_im = sin(angle)

    var w_re = FloatType(1.0)
    var w_im = FloatType(0.0)

    alias sqh = sqrt(0.5).cast[Type]()

    factors_re = List[FloatType](length=n // 2, fill=0.0)
    factors_im = List[FloatType](length=n // 2, fill=0.0)

    for j in range(n // 8):
        factors_re[j] = w_re  # cos(j * angle)
        factors_im[j] = w_im  # sin(j * angle)

        sum = (w_re + w_im) * sqh
        diff = (w_re - w_im) * sqh

        factors_re[j + n // 8] = sum  # cos((j + n // 8) * angle)
        factors_im[j + n // 8] = -diff  # sin((j + n // 8) * angle)

        factors_re[j + n // 4] = w_im  # cos((j + n // 4) * angle)
        factors_im[j + n // 4] = -w_re  # sin((j + n // 4) * angle)

        factors_re[j + 3 * (n // 8)] = -diff  # cos((j + 3 * (n // 8)) * angle)
        factors_im[j + 3 * (n // 8)] = -sum  # sin((j + 3 * (n // 8)) * angle)

        var w_im_new = w_re * wm_im + w_im * wm_re
        var w_re_new = w_re * wm_re - w_im * wm_im

        w_re = w_re_new
        w_im = w_im_new

    return factors_re^, factors_im^


fn fft_dit(mut state: QuantumState, inverse: Bool = False):
    """Apply FFT to the quantum state using Cooley-Tukey algorithm.

    This is an in-place FFT that operates on the state's complex amplitudes.
    Uses the radix-2 decimation-in-time algorithm.

    Args:
        state: QuantumState to transform (modified in-place)
        inverse: If True, performs inverse FFT
    """
    var n = state.size()

    # Bit-reversal permutation
    bit_reverse_state(state)

    ref factors_re, factors_im = generate_factors(n)

    # Cooley-Tukey FFT
    var m = 2
    while m <= n:
        f = n // m
        # Process all groups at this stage
        for k in range(0, n, m):
            # Butterfly operations within group
            for j in range(m // 2):
                w_re = factors_re[f * j]
                w_im = factors_im[f * j]
                var t_re = (
                    w_re * state.re[k + j + m // 2]
                    - w_im * state.im[k + j + m // 2]
                )
                var t_im = (
                    w_re * state.im[k + j + m // 2]
                    + w_im * state.re[k + j + m // 2]
                )

                var u_re = state.re[k + j]
                var u_im = state.im[k + j]

                state.re[k + j] = u_re + t_re
                state.im[k + j] = u_im + t_im
                state.re[k + j + m // 2] = u_re - t_re
                state.im[k + j + m // 2] = u_im - t_im

        m *= 2

    var scale = FloatType(1.0) / sqrt(FloatType(n))
    for i in range(n):
        state.re[i] *= scale
        state.im[i] *= scale


fn fft_dit_simd(mut state: QuantumState, inverse: Bool = False):
    """Vectorized DIT FFT."""
    var n = state.size()
    bit_reverse_state(state)
    ref factors_re, factors_im = generate_factors(n)

    var ptr_re = UnsafePointer(state.re.unsafe_ptr())
    var ptr_im = UnsafePointer(state.im.unsafe_ptr())

    var m = 2
    while m <= n:
        var f = n // m
        var m_half = m // 2

        @parameter
        fn process_simd[width: Int](j: Int):
            # For large strides we can't vectorize factor loading easily unless stride=1
            # But here 'j' is the inner loop index.
            # We want to process 'width' items at j, j+1, ... j+width-1
            # Indices:
            # k is fixed for this block.
            # u_idx = k + j
            # t_idx = k + j + m_half

            # Factors:
            # fac_idx = f * j -> strided!
            # If f == 1 (last stage), stride is 1 (contiguous)
            # If f > 1, we must gather or compute.

            # Since f is power of 2, we can just compute indices?
            # Or use gather.

            # Wait, `process_simd` is called by `vectorize`.
            # `vectorize` passes `j` as scalar index.
            # We assume we are inside the `k` loop.

            # For simple vectorization, let's just use gather/scatter or scalar loads for factors if stride > 1
            pass

        # Strategy:
        # If m_half >= simd_width, we can vectorize the j loop.
        # factor index is f*j. If f==1, it's contiguous. If f>1, it's strided.

        for k in range(0, n, m):
            if m_half >= simd_width:  # Vectorize

                @parameter
                fn inner_simd[width: Int](j: Int):
                    var offset = k + j
                    var offset_t = offset + m_half

                    var u_re = ptr_re.load[width=width](offset)
                    var u_im = ptr_im.load[width=width](offset)
                    var t_re_in = ptr_re.load[width=width](offset_t)
                    var t_im_in = ptr_im.load[width=width](offset_t)

                    # Compute factors
                    var w_re: SIMD[Type, width]
                    var w_im: SIMD[Type, width]

                    if f == 1:
                        w_re = factors_re.unsafe_ptr().load[width=width](f * j)
                        w_im = factors_im.unsafe_ptr().load[width=width](f * j)
                    else:
                        # Manual gather for strided access
                        w_re = SIMD[Type, width]()
                        w_im = SIMD[Type, width]()
                        for s in range(width):
                            w_re[s] = factors_re.unsafe_ptr().load(f * (j + s))
                            w_im[s] = factors_im.unsafe_ptr().load(f * (j + s))

                    # Complex mul: (a+bi)(c+di) = (ac-bd) + i(ad+bc)
                    # t = w * t_in
                    var t_re = w_re * t_re_in - w_im * t_im_in
                    var t_im = w_re * t_im_in + w_im * t_re_in

                    ptr_re.store[width=width](offset, u_re + t_re)
                    ptr_im.store[width=width](offset, u_im + t_im)
                    ptr_re.store[width=width](offset_t, u_re - t_re)
                    ptr_im.store[width=width](offset_t, u_im - t_im)

                vectorize[inner_simd, simd_width](m_half)
            else:
                # Scalar fallback for small blocks
                for j in range(m_half):
                    var w_re = factors_re[f * j]
                    var w_im = factors_im[f * j]

                    var u_re = state.re[k + j]
                    var u_im = state.im[k + j]
                    var t_re_in = state.re[k + j + m // 2]
                    var t_im_in = state.im[k + j + m // 2]

                    var t_re = w_re * t_re_in - w_im * t_im_in
                    var t_im = w_re * t_im_in + w_im * t_re_in

                    state.re[k + j] = u_re + t_re
                    state.im[k + j] = u_im + t_im
                    state.re[k + j + m // 2] = u_re - t_re
                    state.im[k + j + m // 2] = u_im - t_im

        m *= 2

    var scale = FloatType(1.0) / sqrt(FloatType(n))
    # Vectorize normalization
    var ptr_re_norm = UnsafePointer(state.re.unsafe_ptr())
    var ptr_im_norm = UnsafePointer(state.im.unsafe_ptr())

    @parameter
    fn norm_simd[width: Int](i: Int):
        ptr_re_norm.store[width=width](
            i, ptr_re_norm.load[width=width](i) * scale
        )
        ptr_im_norm.store[width=width](
            i, ptr_im_norm.load[width=width](i) * scale
        )

    vectorize[norm_simd, simd_width](n)


fn fft_dif(mut state: QuantumState, inverse: Bool = False):
    """NumPy-style FFT using decimation-in-frequency algorithm."""
    var n = state.size()
    var log_n = Int(log2(Float64(n)))

    # Factors generation (reused or recomputed)
    # We need to inline this properly or share it.
    # For now, we reuse the logic but we need access to factors.
    # The original fft_dif computed factors inside? No, it used generate_factors equivalent logic implicitly?
    # Actually the previous file had a manual copy of factor generation inside generate_factors AND inside fft_dif factors loop.
    # Let's just use generate_factors for cleaner code, assuming cost is negligible or we hoist it.

    ref factors_re, factors_im = generate_factors(n)

    var block_size = n

    for _ in range(log_n):
        var half_block = block_size // 2
        var _ = n // block_size  # Stride for factors?
        # In DIF, we need w^(k * N/block_size)?
        # Original code used: factors_re[n // block_size * j]
        # So stride is n // block_size.

        var stride = n // block_size

        for block_start in range(0, n, block_size):
            for j in range(half_block):
                var top_idx = block_start + j
                var bot_idx = top_idx + half_block

                var top_re = state.re[top_idx]
                var top_im = state.im[top_idx]
                var bot_re = state.re[bot_idx]
                var bot_im = state.im[bot_idx]

                var sum_re = top_re + bot_re
                var sum_im = top_im + bot_im
                var diff_re = top_re - bot_re
                var diff_im = top_im - bot_im

                var w_re = factors_re[stride * j]
                var w_im = factors_im[stride * j]

                var t_re = w_re * diff_re - w_im * diff_im
                var t_im = w_re * diff_im + w_im * diff_re

                # In DIF, we multiply AFTER butterfly on the bottom part?
                # Standard DIF:
                # a = u + v
                # b = (u - v) * w
                # Yes, matching logic above.

                state.re[top_idx] = sum_re
                state.im[top_idx] = sum_im
                state.re[bot_idx] = t_re
                state.im[bot_idx] = t_im

        block_size = half_block

    bit_reverse_state(state)

    var scale = FloatType(1.0) / sqrt(FloatType(n))
    for i in range(n):
        state.re[i] *= scale
        state.im[i] *= scale


fn fft_dif_simd(mut state: QuantumState, inverse: Bool = False):
    """Vectorized DIF FFT."""
    var n = state.size()
    var log_n = Int(log2(Float64(n)))
    ref factors_re, factors_im = generate_factors(n)

    var ptr_re = UnsafePointer(state.re.unsafe_ptr())
    var ptr_im = UnsafePointer(state.im.unsafe_ptr())

    var block_size = n
    for _ in range(log_n):
        var half_block = block_size // 2
        var stride = n // block_size

        for block_start in range(0, n, block_size):
            if half_block >= simd_width:

                @parameter
                fn inner_simd[width: Int](j: Int):
                    var top_idx = block_start + j
                    var bot_idx = top_idx + half_block

                    var top_re = ptr_re.load[width=width](top_idx)
                    var top_im = ptr_im.load[width=width](top_idx)
                    var bot_re = ptr_re.load[width=width](bot_idx)
                    var bot_im = ptr_im.load[width=width](bot_idx)

                    var sum_re = top_re + bot_re
                    var sum_im = top_im + bot_im
                    var diff_re = top_re - bot_re
                    var diff_im = top_im - bot_im

                    # Factors
                    var w_re: SIMD[Type, width]
                    var w_im: SIMD[Type, width]

                    if stride == 1:
                        w_re = factors_re.unsafe_ptr().load[width=width](j)
                        w_im = factors_im.unsafe_ptr().load[width=width](j)
                    else:
                        # Manual gather for strided access
                        w_re = SIMD[Type, width]()
                        w_im = SIMD[Type, width]()
                        for k in range(width):
                            w_re[k] = factors_re.unsafe_ptr().load(
                                stride * (j + k)
                            )
                            w_im[k] = factors_im.unsafe_ptr().load(
                                stride * (j + k)
                            )

                    # t = diff * w
                    var t_re = diff_re * w_re - diff_im * w_im
                    var t_im = diff_re * w_im + diff_im * w_re

                    ptr_re.store[width=width](top_idx, sum_re)
                    ptr_im.store[width=width](top_idx, sum_im)
                    ptr_re.store[width=width](bot_idx, t_re)
                    ptr_im.store[width=width](bot_idx, t_im)

                vectorize[inner_simd, simd_width](half_block)
            else:
                # Scalar fallback
                for j in range(half_block):
                    var w_re = factors_re[stride * j]
                    var w_im = factors_im[stride * j]

                    var top_idx = block_start + j
                    var bot_idx = top_idx + half_block

                    var top_re = state.re[top_idx]
                    var top_im = state.im[top_idx]
                    var bot_re = state.re[bot_idx]
                    var bot_im = state.im[bot_idx]

                    var sum_re = top_re + bot_re
                    var sum_im = top_im + bot_im
                    var diff_re = top_re - bot_re
                    var diff_im = top_im - bot_im

                    var t_re = diff_re * w_re - diff_im * w_im
                    var t_im = diff_re * w_im + diff_im * w_re

                    state.re[top_idx] = sum_re
                    state.im[top_idx] = sum_im
                    state.re[bot_idx] = t_re
                    state.im[bot_idx] = t_im

        block_size = half_block

    bit_reverse_state(state)

    var scale = FloatType(1.0) / sqrt(FloatType(n))
    var ptr_re_norm = UnsafePointer(state.re.unsafe_ptr())
    var ptr_im_norm = UnsafePointer(state.im.unsafe_ptr())

    @parameter
    fn norm_simd[width: Int](i: Int):
        ptr_re_norm.store[width=width](
            i, ptr_re_norm.load[width=width](i) * scale
        )
        ptr_im_norm.store[width=width](
            i, ptr_im_norm.load[width=width](i) * scale
        )

    vectorize[norm_simd, simd_width](n)


fn fft_dit_parallel(mut state: QuantumState, inverse: Bool = False):
    """Parallelized DIT FFT."""
    var n = state.size()
    bit_reverse_state(state)
    ref factors_re, factors_im = generate_factors(n)

    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()
    var ptr_fac_re = factors_re.unsafe_ptr()
    var ptr_fac_im = factors_im.unsafe_ptr()

    var m = 2
    while m <= n:
        var f = n // m
        var m_half = m // 2

        @parameter
        fn stage_worker(idx: Int):
            # idx goes from 0 to n/2 - 1
            var group = idx // m_half
            var j = idx % m_half
            var k = group * m

            var w_re = ptr_fac_re[f * j]
            var w_im = ptr_fac_im[f * j]

            var u_re = ptr_re[k + j]
            var u_im = ptr_im[k + j]
            var t_re_in = ptr_re[k + j + m_half]
            var t_im_in = ptr_im[k + j + m_half]

            var t_re = w_re * t_re_in - w_im * t_im_in
            var t_im = w_re * t_im_in + w_im * t_re_in

            ptr_re[k + j] = u_re + t_re
            ptr_im[k + j] = u_im + t_im
            ptr_re[k + j + m_half] = u_re - t_re
            ptr_im[k + j + m_half] = u_im - t_im

        parallelize[stage_worker](n // 2, n // 2)
        m *= 2

    var scale = FloatType(1.0) / sqrt(FloatType(n))

    @parameter
    fn norm_worker(i: Int):
        ptr_re[i] *= scale
        ptr_im[i] *= scale

    parallelize[norm_worker](n)


fn fft_dif_parallel(mut state: QuantumState, inverse: Bool = False):
    """Parallelized DIF FFT."""
    var n = state.size()
    var log_n = Int(log2(Float64(n)))
    ref factors_re, factors_im = generate_factors(n)

    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()
    var ptr_fac_re = factors_re.unsafe_ptr()
    var ptr_fac_im = factors_im.unsafe_ptr()

    var stride = n // 2
    for _ in range(log_n):

        @parameter
        fn stage_worker(idx: Int):
            # idx: 0 to n/2 - 1
            var block_idx = idx // stride
            var block_start = block_idx * 2 * stride
            var j = idx % stride

            var top_idx = block_start + j
            var bot_idx = top_idx + stride

            var top_re = ptr_re[top_idx]
            var top_im = ptr_im[top_idx]
            var bot_re = ptr_re[bot_idx]
            var bot_im = ptr_im[bot_idx]

            var sum_re = top_re + bot_re
            var sum_im = top_im + bot_im
            var diff_re = top_re - bot_re
            var diff_im = top_im - bot_im

            var w_re = ptr_fac_re[n // 2 // stride * j]
            var w_im = ptr_fac_im[n // 2 // stride * j]

            var t_re = diff_re * w_re - diff_im * w_im
            var t_im = diff_re * w_im + diff_im * w_re

            ptr_re[top_idx] = sum_re
            ptr_im[top_idx] = sum_im
            ptr_re[bot_idx] = t_re
            ptr_im[bot_idx] = t_im

        parallelize[stage_worker](n // 2, 8)
        stride = stride // 2

    bit_reverse_state(state)

    var scale = FloatType(1.0) / sqrt(FloatType(n))
    # State was replaced in bit_reverse_state, so we need fresh pointers
    var final_ptr_re = state.re.unsafe_ptr()
    var final_ptr_im = state.im.unsafe_ptr()

    @parameter
    fn norm_worker(i: Int):
        final_ptr_re[i] *= scale
        final_ptr_im[i] *= scale

    parallelize[norm_worker](n)
