import benchmark
from python import Python
from butterfly.core.state import generate_state, QuantumState
from butterfly.core.classical_fft import fft_dif_parallel_simd_phast
from butterfly.core.fft_v3 import fft_v3


fn main() raises:
    var pyfftw = Python.import_module("pyfftw")
    var np = Python.import_module("numpy")
    var time = Python.import_module("time")

    # Configure pyfftw to use the cache to be faster
    _ = pyfftw.interfaces.cache.enable()
    _ = pyfftw.config.NUM_THREADS = 10

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

        # Run Both
        # fft_dif_parallel_simd_phast(v_state)
        fft_v3(v_state, block_log=18)
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

    print(
        "n, Size, Butterfly(ms), V3(ms), FFTW(ms), Speedup(V3/FFTW),"
        " Speedup(Butterfly/FFTW)"
    )

    for n in [15, 20, 25]:  # Benchmarking High N
        var size = 1 << n
        var state = generate_state(n)

        # Iterations
        var iters = 5 if n < 21 else 2

        var state_butterfly = state.copy()
        var state_v3 = state.copy()

        # FFTW setup
        # Use empty_aligned for optimal FFTW performance
        var fftw_in = pyfftw.empty_aligned(size, dtype="complex128")
        fftw_in.real = np.random.rand(size)
        fftw_in.imag = np.random.rand(size)

        # Warmup
        fft_dif_parallel_simd_phast(state_butterfly)
        fft_v3(state_butterfly, block_log=18)
        # _ = pyfftw.interfaces.numpy_fft.fft(fftw_in)
        var fftw_obj = pyfftw.builders.fft(
            fftw_in, planner_effort="FFTW_MEASURE"
        )

        # 3. Butterfly
        var t_butterfly_0 = time.time()
        for _ in range(iters):
            fft_dif_parallel_simd_phast(state_butterfly)
        var t_butterfly_1 = time.time()
        var dur_butterfly = Float64(
            (t_butterfly_1 - t_butterfly_0) * 1000.0 / iters
        )

        # V3
        var t_v3_0 = time.time()
        for _ in range(iters):
            fft_v3(state_v3, block_log=18)
        var t_v3_1 = time.time()
        var dur_v3 = Float64((t_v3_1 - t_v3_0) * 1000.0 / iters)

        # 4. FFTW
        var t4 = time.time()
        for _ in range(iters):
            # _ = pyfftw.interfaces.numpy_fft.fft(fftw_in)
            _ = fftw_obj()
        var t5 = time.time()
        var dur_fftw = Float64((t5 - t4) * 1000.0 / iters)

        print(
            n,
            ", ",
            size,
            ", ",
            dur_butterfly,
            ", ",
            dur_v3,
            ", ",
            dur_fftw,
            ", ",
            dur_v3 / dur_fftw,
            ", ",
            dur_butterfly / dur_fftw,
        )
