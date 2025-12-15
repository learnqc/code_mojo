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
    print(
        "n, Size, SIMD(ms), FastDiv(ms), PhastFT(ms), NumPy(ms),"
        " Speedup(Phast/NumPy)"
    )

    for n in range(23, 27):  # Benchmarking High N
        var size = 1 << n
        var state = generate_state(n)

        # Iterations: vary based on size strictly
        var iters = 3 if n < 23 else 2

        var state_simd = state
        var state_simd_fast = state
        var state_phast = state

        # NumPy setup
        var np_in = np.zeros(size, dtype=np.complex128)
        np_in.real = np.random.rand(size)
        np_in.imag = np.random.rand(size)

        # Warmup (skip for largest sizes to save time?)
        # Just 1 run warmup is fine
        fft_dif_parallel_simd(state_simd)
        fft_dif_parallel_simd_fastdiv(state_simd_fast)
        fft_dif_parallel_simd_phast(state_phast)
        _ = np.fft.fft(np_in)

        # 1. SIMD (Std)
        var t0 = time.time()
        for _ in range(iters):
            fft_dif_parallel_simd(state_simd)
        var t1 = time.time()
        var dur_simd = Float64((t1 - t0) * 1000.0 / iters)

        # 2. SIMD (FastDiv Optimization)
        var t2 = time.time()
        for _ in range(iters):
            fft_dif_parallel_simd_fastdiv(state_simd_fast)
        var t3 = time.time()
        var dur_simd_fast = Float64((t3 - t2) * 1000.0 / iters)

        # 3. PhastFT (Tiled Optimization)
        var t_phast_0 = time.time()
        for _ in range(iters):
            fft_dif_parallel_simd_phast(state_phast)
        var t_phast_1 = time.time()
        var dur_phast = Float64((t_phast_1 - t_phast_0) * 1000.0 / iters)

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
            dur_simd,
            ", ",
            dur_simd_fast,
            ", ",
            dur_phast,
            ", ",
            dur_np,
            ", ",
            dur_np / dur_phast,  # Speedup of Phast vs NumPy
        )
