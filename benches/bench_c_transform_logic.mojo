import benchmark


fn is_bit_set(m: Int, k: Int) -> Bool:
    return (m & (1 << k)) != 0


fn c_transform_logic(n: Int, control: Int, target: Int) -> Int:
    var count = 0
    var stride = 1 << target
    var size = 1 << n
    var r = 0
    for j in range(size // 2):
        var idx = 2 * j - r
        if is_bit_set(Int(idx), control):
            count += 1
        r += 1
        if r == stride:
            r = 0
    return count


fn c_transform_interval_logic(n: Int, control: Int, target: Int) -> Int:
    var count = 0
    var c_stride = 1 << control
    var t_stride = 1 << target
    var size = 1 << n

    if target < control:
        for k in range(size // (2 * c_stride)):
            var block_start = k * 2 * c_stride + c_stride
            for j in range(c_stride // (2 * t_stride)):
                var sub_start = block_start + j * 2 * t_stride
                for idx in range(sub_start, sub_start + t_stride):
                    count += 1
    else:
        for k in range(size // (2 * t_stride)):
            var base = k * 2 * t_stride
            var num_periods = t_stride // (2 * c_stride)
            for p in range(num_periods):
                var p_start = base + p * 2 * c_stride + c_stride
                for idx in range(p_start, p_start + c_stride):
                    count += 1
    return count


# Wrappers for benchmarking
fn bench_case_1_original():
    benchmark.keep(c_transform_logic(20, 10, 0))


fn bench_case_1_interval():
    benchmark.keep(c_transform_interval_logic(20, 10, 0))


fn bench_case_2_original():
    benchmark.keep(c_transform_logic(20, 0, 10))


fn bench_case_2_interval():
    benchmark.keep(c_transform_interval_logic(20, 0, 10))


fn bench_case_3_original():
    benchmark.keep(c_transform_logic(20, 10, 9))


fn bench_case_3_interval():
    benchmark.keep(c_transform_interval_logic(20, 10, 9))


fn main() raises:
    print("Benchmarking N=20...")

    print("Case 1: Control=10, Target=0")
    var report1_orig = benchmark.run[bench_case_1_original](5, 10)
    var report1_new = benchmark.run[bench_case_1_interval](5, 10)
    print("Original:", report1_orig.mean(benchmark.Unit.ns), "ns")
    print("Interval:", report1_new.mean(benchmark.Unit.ns), "ns")

    print("Case 2: Control=0, Target=10")
    var report2_orig = benchmark.run[bench_case_2_original](5, 10)
    var report2_new = benchmark.run[bench_case_2_interval](5, 10)
    print("Original:", report2_orig.mean(benchmark.Unit.ns), "ns")
    print("Interval:", report2_new.mean(benchmark.Unit.ns), "ns")

    print("Case 3: Control=10, Target=9 (Close)")
    var report3_orig = benchmark.run[bench_case_3_original](5, 10)
    var report3_new = benchmark.run[bench_case_3_interval](5, 10)
    print("Original:", report3_orig.mean(benchmark.Unit.ns), "ns")
    print("Interval:", report3_new.mean(benchmark.Unit.ns), "ns")
