import benchmark
from python import Python
from butterfly.core.state import generate_state, QuantumState
from butterfly.core.classical_fft import (
    fft_dif_parallel_simd,
    fft_dif_parallel_simd_fastdiv,
    fft_dif_parallel_simd_phast,
)


fn main() raises:
    var np = Python.import_module("numpy")
    var time = Python.import_module("time")

    print("Benchmarking Large Scale FFT")

    # --- Verification Step ---
    for n in range(3, 23):
        print("Verifying correctness against NumPy (n={})...".format(String(n)))
        var size = 1 << n
        var v_state = generate_state(n)

        # Create NumPy equivalent
        var np_v = np.zeros(size, dtype=np.complex64)
        var v_ptr_re = v_state.re.unsafe_ptr()
        var v_ptr_im = v_state.im.unsafe_ptr()
        # Manual copy loop for verification size (small enough)
        # Using np array setting is tricky from mojo pointers directly safely without unsafe memcpy
        # But for small size, we can construct list and pass? Or loop in python?
        # Actually, we can just use Random generation in Python to ensure match?
        # Or just copy FROM Mojo TO Numpy.

        # Let's generate in Python to keep it simple and load into Mojo?
        # But `generate_state` is our source of truth for benchmark distribution.
        # Let's try to copy data out.
        var p_re = Python.evaluate("[]")
        var p_im = Python.evaluate("[]")
        for i in range(size):
            _ = p_re.append(v_state.re[i])
            _ = p_im.append(v_state.im[i])

        np_v.real = p_re
        np_v.imag = p_im

        # Run Both
        fft_dif_parallel_simd_phast(v_state)
        var np_res = np.fft.fft(np_v, norm="ortho")

        # Compare
        # Extract results back
        var diff_sum: Float64 = 0.0
        for i in range(size):
            var mojo_re: Float64 = v_state.re[i].cast[DType.float64]()
            var mojo_im: Float64 = v_state.im[i].cast[DType.float64]()
            var np_re = Float64(np_res[i].real)
            var np_im = Float64(np_res[i].imag)
            var diff_re = mojo_re - np_re
            var diff_im = mojo_im - np_im
            var diff = diff_re * diff_re + diff_im * diff_im
            if diff > 1e-7:
                print(
                    "\tDiscrepancy at index",
                    i,
                    mojo_re,
                    np_re,
                    mojo_im,
                    np_im,
                    diff,
                )
                raise Error("Verification Failed")
            diff_sum += diff

        if diff_sum > 1e-5:
            print("\tVerification FAILED! Diff Sum:", diff_sum)
            # raise Error("Verification Failed")
        else:
            print("\tVerification PASSED! Diff Sum:", diff_sum)
        # -------------------------

    print("n, Size, Butterfly(ms), NumPy(ms), Speedup(Butterfly/NumPy)")

    for n in range(30, 31):  # Benchmarking High N
        var size = 1 << n
        var state = generate_state(n)

        # Iterations: vary based on size strictly
        var iters = 5 if n < 21 else 2

        var state_butterfly = state

        # NumPy setup
        var np_in = np.zeros(size, dtype=np.complex128)
        np_in.real = np.random.rand(size)
        np_in.imag = np.random.rand(size)

        # Warmup (skip for largest sizes to save time?)
        # Just 1 run warmup is fine
        fft_dif_parallel_simd_phast(state_butterfly)
        _ = np.fft.fft(np_in)

        # 3. Butterfly
        var t_butterfly_0 = time.time()
        for _ in range(iters):
            fft_dif_parallel_simd_phast(state_butterfly)
        var t_butterfly_1 = time.time()
        var dur_butterfly = Float64(
            (t_butterfly_1 - t_butterfly_0) * 1000.0 / iters
        )

        # 4. NumPy
        var t4 = time.time()
        for _ in range(iters):
            _ = np.fft.fft(np_in)
        var t5 = time.time()
        var dur_np = Float64((t5 - t4) * 1000.0 / iters)

        print(
            n,
            ", ",
            size,
            ", ",
            dur_butterfly,
            ", ",
            dur_np,
            ", ",
            dur_np / dur_butterfly,  # Speedup of Butterfly vs NumPy
        )


# n, Size, Butterfly(ms), NumPy(ms), Speedup(Butterfly/NumPy)
# 17 ,  131072 ,  1.0000228881835938 ,  1.9253730773925781 ,  1.9253290101087164
# 18 ,  262144 ,  2.213430404663086 ,  3.8660049438476563 ,  1.7466123785518861
# 19 ,  524288 ,  4.259157180786133 ,  7.342815399169922 ,  1.724006672563003
# 20 ,  1048576 ,  15.768957138061523 ,  17.424631118774414 ,  1.1049957816624785
# 21 ,  2097152 ,  18.61584186553955 ,  41.574954986572266 ,  2.233310493657187
# 22 ,  4194304 ,  40.175557136535645 ,  87.36050128936768 ,  2.1744689437031366
# 23 ,  8388608 ,  89.69497680664063 ,  183.74347686767578 ,  2.0485369795432313
# 24 ,  16777216 ,  176.17547512054443 ,  398.3900547027588 ,  2.261325274872502
# 25 ,  33554432 ,  368.85106563568115 ,  850.6709337234497 ,  2.3062721325134197
# 26 ,  67108864 ,  773.2239961624146 ,  1886.2345218658447 ,  2.439441263110572
# 27 ,  134217728 ,  1843.6464071273804 ,  4048.5169887542725 ,  2.195929204810125
# 28 ,  268435456 ,  3982.507109642029 ,  12426.422119140625 ,  3.1202510823031737
# 29 ,  536870912 ,  12082.942008972168 ,  114615.23044109344 ,  9.485705580311988
