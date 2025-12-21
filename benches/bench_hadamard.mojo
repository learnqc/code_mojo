import benchmark
from python import Python
from butterfly.core.state import QuantumState, generate_state
from butterfly.algos.hadamard import hadamard_transform
from butterfly.core.types import FloatType
from math import sqrt


fn main() raises:
    print("Benchmarking Recursive Hadamard (WHT)...")
    var time_py = Python.import_module("time")
    var np = Python.import_module("numpy")

    # 1. Verification (N=10, Size=1024)
    # H * H = I
    var verify_n = 10
    var verify_size = 1 << verify_n
    var s = QuantumState(verify_n)

    # Random inputs
    var data = np.random.rand(verify_size).astype(np.float64)
    for i in range(verify_size):
        var val_py = data[i].item()
        s.re[i] = FloatType(Float64(val_py))
        s.im[i] = 0.0

    # Copy for check
    var original = List[FloatType]()
    for i in range(verify_size):
        original.append(s.re[i])

    # Forward H
    hadamard_transform(s)

    # Inverse H (Same function!)
    hadamard_transform(s)

    # Check
    var max_err = 0.0
    for i in range(verify_size):
        var diff = abs(s.re[i] - original[i])
        if diff > max_err:
            max_err = diff

    print("Verification (N=10) Max Error:", max_err)
    if max_err < 1e-9:
        print("PASS: Round trip successful.")
    else:
        print("FAIL: Round trip error too high.")

    # 2. Benchmark
    print("\nPerformance Benchmark:")
    print("N, Size, Time(ms)")

    for n in range(16, 25):
        var size = 1 << n
        var s_bench = QuantumState(n)

        # Init
        for i in range(size):
            s_bench.re[i] = 1.0
            s_bench.im[i] = 0.0

        # Warmup
        hadamard_transform(s_bench)

        var t0 = time_py.time()
        var runs = 5
        for _ in range(runs):
            hadamard_transform(s_bench)
        var t1 = time_py.time()

        # Convert times individually to avoid PythonObject arithmetic ambiguity
        var start_s = Float64(t0)
        var end_s = Float64(t1)
        var avg_ms = (end_s - start_s) / runs * 1000.0
        print(n, ",", size, ",", avg_ms)
