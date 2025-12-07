import benchmark
from butterfly.core.state import (
    QuantumState,
    c_transform_simd,
    c_transform_interval_simd,
)
from butterfly.core.gates import X


# Aliasing N to a constant for templated functions
alias N = 20


fn bench_case_1_simd():
    var state = QuantumState(N)
    c_transform_simd[N](state, 10, 0, X)
    benchmark.keep(state.re[0])


fn bench_case_1_interval_simd():
    var state = QuantumState(N)
    c_transform_interval_simd[N](state, 10, 0, X)
    benchmark.keep(state.re[0])


fn bench_case_2_simd():
    var state = QuantumState(N)
    c_transform_simd[N](state, 0, 10, X)
    benchmark.keep(state.re[0])


fn bench_case_2_interval_simd():
    var state = QuantumState(N)
    c_transform_interval_simd[N](state, 0, 10, X)
    benchmark.keep(state.re[0])


fn bench_case_3_simd():
    var state = QuantumState(N)
    c_transform_simd[N](state, 10, 9, X)
    benchmark.keep(state.re[0])


fn bench_case_3_interval_simd():
    var state = QuantumState(N)
    c_transform_interval_simd[N](state, 10, 9, X)
    benchmark.keep(state.re[0])


fn main() raises:
    print("Benchmarking SIMD N=20...")

    print("Case 1: Control=10, Target=0 (Target < Control, strides 1 vs 1024)")
    var report1_simd = benchmark.run[bench_case_1_simd](5, 1000)
    var report1_int_simd = benchmark.run[bench_case_1_interval_simd](5, 1000)
    print("SIMD Standard:    ", report1_simd.mean(benchmark.Unit.ms), "ms")
    print("SIMD Interval:    ", report1_int_simd.mean(benchmark.Unit.ms), "ms")

    print(
        "\nCase 2: Control=0, Target=10 (Target > Control, strides 1024 vs 1)"
    )
    var report2_simd = benchmark.run[bench_case_2_simd](5, 1000)
    var report2_int_simd = benchmark.run[bench_case_2_interval_simd](5, 1000)
    print("SIMD Standard:    ", report2_simd.mean(benchmark.Unit.ms), "ms")
    print("SIMD Interval:    ", report2_int_simd.mean(benchmark.Unit.ms), "ms")

    print("\nCase 3: Control=10, Target=9 (Close, strides 512 vs 1024)")
    var report3_simd = benchmark.run[bench_case_3_simd](5, 1000)
    var report3_int_simd = benchmark.run[bench_case_3_interval_simd](5, 1000)
    print("SIMD Standard:    ", report3_simd.mean(benchmark.Unit.ms), "ms")
    print("SIMD Interval:    ", report3_int_simd.mean(benchmark.Unit.ms), "ms")
