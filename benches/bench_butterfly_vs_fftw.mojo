import benchmark
from python import Python
from butterfly.core.state import generate_state, QuantumState
from butterfly.core.classical_fft import (
    fft_dif_parallel_simd_phast,
    fft_dif_parallel_simd_phast_kernel,
    generate_factors,
)
from butterfly.core.fft_v3 import fft_v3, fft_v3_kernel
from butterfly.core.fft_v4 import fft_v4, fft_v4_kernel
from butterfly.core.fft_v4_plus import fft_v4_plus, fft_v4_plus_kernel
from butterfly.core.fft_v4_opt import fft_v4_opt, fft_v4_opt_kernel


fn main() raises:
    var pre_compute_twiddles = False
    var pyfftw = Python.import_module("pyfftw")
    var np = Python.import_module("numpy")
    var time = Python.import_module("time")

    # Configure pyfftw to use the cache to be faster
    _ = pyfftw.interfaces.cache.enable()
    _ = pyfftw.config.NUM_THREADS = 8

    print("Benchmarking Butterfly vs FFTW")

    # --- Verification Step ---
    for n in range(3, 9):
        print("Verifying correctness against FFTW (n={})...".format(String(n)))
        var size = 1 << n
        var v_state = generate_state(n)

        # Create NumPy/FFTW equivalent
        var np_v = pyfftw.empty_aligned(size, dtype="complex64")

        var p_re = Python.evaluate("[]")
        var p_im = Python.evaluate("[]")
        for i in range(size):
            _ = p_re.append(v_state.re[i])
            _ = p_im.append(v_state.im[i])

        np_v.real = p_re
        np_v.imag = p_im

        # Run Both (test V4 Plus)
        # fft_v4_plus(v_state, block_log=12)
        fft_v4_opt(v_state, block_log=12)
        # Using pyfftw.interfaces.numpy_fft.fft which is a drop-in replacement
        var fftw_res = pyfftw.interfaces.numpy_fft.fft(np_v, norm="ortho")

        # Compare
        var diff_sum: Float64 = 0.0
        for i in range(size):
            var mojo_re: Float64 = v_state.re[i].cast[DType.float64]()
            var mojo_im: Float64 = v_state.im[i].cast[DType.float64]()
            var fftw_re = Float64(fftw_res[i].real)
            var fftw_im = Float64(fftw_res[i].imag)
            var diff_re = mojo_re - fftw_re
            var diff_im = mojo_im - fftw_im
            var diff = diff_re * diff_re + diff_im * diff_im
            if diff > 1e-7:
                print(
                    "\tDiscrepancy at index",
                    i,
                    mojo_re,
                    fftw_re,
                    mojo_im,
                    fftw_im,
                    diff,
                )
                raise Error("Verification Failed")
            diff_sum += diff

        if diff_sum > 1e-5:
            print("\tVerification FAILED! Diff Sum:", diff_sum)
        else:
            print("\tVerification PASSED! Diff Sum:", diff_sum)
        # -------------------------

    var row_fmt = "{:<3} | {:<10} | {:<10.2f} | {:<10.2f} | {:<10.2f} | {:<10.2f} | {:<10.2f} | {:<10.2f} | {:<10.2f}"
    var header_fmt = "{:<3} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10}"
    var py_print = Python.evaluate(
        "lambda fmt, *args: print(fmt.format(*args))"
    )

    py_print(
        header_fmt,
        "n",
        "Size",
        "Phast(ms)",
        "V3(ms)",
        "V4(ms)",
        "V4 Plus(ms)",
        "V4 Opt(ms)",
        "FFTW(ms)",
        "% FFTW",
    )
    print("-" * 120)

    for n in range(3, 29):  # Benchmarking High N
        var size = 1 << n
        var iters = 5 if n < 21 else 2

        # Setup states
        var state = generate_state(n)
        var state_butterfly = state.copy()
        var state_v3 = state.copy()
        var state_v4 = state.copy()
        var state_v4_plus = state.copy()
        var state_v4_opt = state.copy()

        # FFTW setup
        var fftw_in = pyfftw.empty_aligned(size, dtype="complex128")
        fftw_in.real = np.random.rand(size)
        fftw_in.imag = np.random.rand(size)

        var fftw_obj = Python.evaluate("None")
        if pre_compute_twiddles:
            fftw_obj = pyfftw.builders.fft(
                fftw_in, planner_effort="FFTW_MEASURE"
            )

        # Pre-compute factors once
        ref factors_re, factors_im = generate_factors(size)

        # Warmup
        fft_dif_parallel_simd_phast_kernel(
            state_butterfly, factors_re, factors_im
        )
        fft_v3_kernel(state_v3, block_log=18)
        fft_v4_kernel(state_v4, factors_re, factors_im, block_log=12)
        fft_v4_plus_kernel(state_v4_plus, factors_re, factors_im, block_log=12)
        fft_v4_opt_kernel(state_v4_opt, factors_re, factors_im, block_log=12)

        # 1. Phast Kernel
        var t_butterfly_0 = time.time()
        for _ in range(iters):
            if not pre_compute_twiddles:
                ref f_re, f_im = generate_factors(size)
                fft_dif_parallel_simd_phast_kernel(state_butterfly, f_re, f_im)
            else:
                fft_dif_parallel_simd_phast_kernel(
                    state_butterfly, factors_re, factors_im
                )
        var t_butterfly_1 = time.time()
        var dur_butterfly = Float64(
            (t_butterfly_1 - t_butterfly_0) * 1000.0 / iters
        )

        # 2. V3 Kernel
        var t_v3_0 = time.time()
        for _ in range(iters):
            fft_v3_kernel(state_v3, block_log=18)
        var t_v3_1 = time.time()
        var dur_v3 = Float64((t_v3_1 - t_v3_0) * 1000.0 / iters)

        # 3. V4 Kernel
        var t_v4_0 = time.time()
        for _ in range(iters):
            if not pre_compute_twiddles:
                ref f_re, f_im = generate_factors(size)
                fft_v4_kernel(state_v4, f_re, f_im, block_log=12)
            else:
                fft_v4_kernel(state_v4, factors_re, factors_im, block_log=12)
        var t_v4_1 = time.time()
        var dur_v4 = Float64((t_v4_1 - t_v4_0) * 1000.0 / iters)

        # 4. V4 Plus Kernel
        var t_v4_plus_0 = time.time()
        for _ in range(iters):
            if not pre_compute_twiddles:
                ref f_re, f_im = generate_factors(size)
                fft_v4_plus_kernel(state_v4_plus, f_re, f_im, block_log=12)
            else:
                fft_v4_plus_kernel(
                    state_v4_plus, factors_re, factors_im, block_log=12
                )
        var t_v4_plus_1 = time.time()
        var dur_v4_plus = Float64((t_v4_plus_1 - t_v4_plus_0) * 1000.0 / iters)

        # 5. V4 Opt Kernel
        var t_v4_opt_0 = time.time()
        for _ in range(iters):
            if not pre_compute_twiddles:
                ref f_re, f_im = generate_factors(size)
                fft_v4_opt_kernel(state_v4_opt, f_re, f_im, block_log=12)
            else:
                fft_v4_opt_kernel(
                    state_v4_opt, factors_re, factors_im, block_log=12
                )
        var t_v4_opt_1 = time.time()
        var dur_v4_opt = Float64((t_v4_opt_1 - t_v4_opt_0) * 1000.0 / iters)

        # 5. FFTW (Planned vs Default)
        var t4 = time.time()
        for _ in range(iters):
            if pre_compute_twiddles:
                _ = fftw_obj()
            else:
                _ = pyfftw.interfaces.numpy_fft.fft(fftw_in)
        var t5 = time.time()
        var dur_fftw = Float64((t5 - t4) * 1000.0 / iters)

        py_print(
            row_fmt,
            n,
            size,
            dur_butterfly,
            dur_v3,
            dur_v4,
            dur_v4_plus,
            dur_v4_opt,
            dur_fftw,
            (dur_fftw / dur_v4_opt) * 100,
            " %",
        )
