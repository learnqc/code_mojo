import benchmark
from butterfly.core.state import QuantumState, c_transform, c_transform_interval
from butterfly.core.gates import X


fn bench_case_1_original():
    var state = QuantumState(20)
    c_transform(state, 10, 0, X)
    benchmark.keep(state.re[0])


fn bench_case_1_interval():
    var state = QuantumState(20)
    c_transform_interval(state, 10, 0, X)
    benchmark.keep(state.re[0])


fn bench_case_2_original():
    var state = QuantumState(20)
    c_transform(state, 0, 10, X)
    benchmark.keep(state.re[0])


fn bench_case_2_interval():
    var state = QuantumState(20)
    c_transform_interval(state, 0, 10, X)
    benchmark.keep(state.re[0])


fn bench_case_3_original():
    var state = QuantumState(20)
    c_transform(state, 10, 9, X)
    benchmark.keep(state.re[0])


fn bench_case_3_interval():
    var state = QuantumState(20)
    c_transform_interval(state, 10, 9, X)
    benchmark.keep(state.re[0])


fn main() raises:
    print("Benchmarking N=20...")

    print("Case 1: Control=10, Target=0")
    var report1_orig = benchmark.run[bench_case_1_original](5, 1000)
    var report1_new = benchmark.run[bench_case_1_interval](5, 100)
    print("Original:", report1_orig.mean(benchmark.Unit.ms), "ms")
    print("Interval:", report1_new.mean(benchmark.Unit.ms), "ms")

    print("Case 2: Control=0, Target=10")
    var report2_orig = benchmark.run[bench_case_2_original](5, 1000)
    var report2_new = benchmark.run[bench_case_2_interval](5, 1000)
    print("Original:", report2_orig.mean(benchmark.Unit.ms), "ms")
    print("Interval:", report2_new.mean(benchmark.Unit.ms), "ms")

    print("Case 3: Control=10, Target=9 (Close)")
    var report3_orig = benchmark.run[bench_case_3_original](5, 1000)
    var report3_new = benchmark.run[bench_case_3_interval](5, 1000)
    print("Original:", report3_orig.mean(benchmark.Unit.ms), "ms")
    print("Interval:", report3_new.mean(benchmark.Unit.ms), "ms")


# Benchmarking N=20...
# Case 1: Control=10, Target=0
# Original: 0.9657838067879562 ms
# Interval: 0.8454805194805195 ms
# Case 2: Control=0, Target=10
# Original: 1.3567708695652172 ms
# Interval: 0.8791083676268862 ms
# Case 3: Control=10, Target=9 (Close)
# Original: 1.3205434782608696 ms
# Interval: 0.8690327485380117 ms
