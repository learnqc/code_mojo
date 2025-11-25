from math import sqrt, log2
from sys.info import simd_width_of

from complex import ComplexSIMD

from algorithm import elementwise, vectorize, parallelize
from buffer import NDBuffer
from testing import assert_almost_equal

from utils.index import IndexList
from utils.variant import Variant

import benchmark

alias dtype = DType.float32
alias float_type = Scalar[dtype]

alias Amplitude = ComplexSIMD[dtype, 1]
alias sq2: Amplitude = Amplitude(sqrt(0.5).cast[dtype](), 0)

alias simd_width = simd_width_of[dtype]()

alias simd_type = Variant[Int, Bool]

alias Gate = InlineArray[InlineArray[Amplitude, 2], 2]
alias FloatGate = InlineArray[InlineArray[Amplitude, 2], 2]


# alias sq2_float = sqrt(0.5).cast[dtype]()
alias H: Gate = [[sq2, sq2], [sq2, -sq2]]


alias X: Gate = [[Amplitude(sqrt(0.0).cast[dtype](), 0.0), Amplitude(sqrt(1.0).cast[dtype](), 0)],
    [Amplitude(sqrt(1.0).cast[dtype](), 0.0), Amplitude(sqrt(0.0).cast[dtype](), 0)]]


fn transform[N: Int, use_vectorize: simd_type = 0, show: Bool=False](mut re: List[float_type], mut im: List[float_type],
    gate: Gate, stride: Int):

    gate_re = [[gate[0][0].re, gate[0][1].re], [gate[1][0].re, gate[1][1].re]]
    gate_im = [[gate[0][0].im, gate[0][1].im], [gate[1][0].im, gate[1][1].im]]

    alias num_work_items = 8
    alias num_threads = num_work_items
    alias chunk_size = N//2//num_work_items

    var vector_re = NDBuffer[dtype, 1, _, N](re)
    var vector_im = NDBuffer[dtype, 1, _, N](im)


    @always_inline
    @parameter
    fn butterfly_simd[simd_width: Int](idx: Int):
        zero_idx = 2*idx - idx%stride
        one_idx = zero_idx + stride
        if show:
            print(idx, ", pairs:", zero_idx, "--", zero_idx+simd_width, one_idx, "--", one_idx+simd_width)

        var elem0_re = vector_re.load[width=simd_width](zero_idx)
        var elem0_im = vector_im.load[width=simd_width](zero_idx)
        var elem1_re = vector_re.load[width=simd_width](one_idx)
        var elem1_im = vector_im.load[width=simd_width](one_idx)

        elem0_orig_re = elem0_re
        elem0_orig_im = elem0_im

        elem1_orig_re = elem1_re
        elem1_orig_im = elem1_im

        elem0_re = elem0_orig_re.fma(gate_re[0][0], -gate_im[0][0]*elem0_orig_im + elem1_orig_re.fma(gate_re[0][1], -gate_im[0][1]*elem1_orig_im))
        elem0_im = elem0_orig_re.fma(gate_im[0][0], gate_re[0][0]*elem0_orig_im + elem1_orig_re.fma(gate_im[0][1], gate_re[0][1]*elem1_orig_im))
        elem1_re = elem0_orig_re.fma(gate_re[1][0], -gate_im[1][0]*elem0_orig_im + elem1_orig_re.fma(gate_re[1][1], -gate_im[1][1]*elem1_orig_im))
        elem1_im = elem0_orig_re.fma(gate_im[1][0], gate_re[1][0]*elem0_orig_im + elem1_orig_re.fma(gate_im[1][1], gate_re[1][1]*elem1_orig_im))

        vector_re.store[width=simd_width](zero_idx, elem0_re)
        vector_im.store[width=simd_width](zero_idx, elem0_im)
        vector_re.store[width=simd_width](one_idx, elem1_re)
        vector_im.store[width=simd_width](one_idx, elem1_im)

    @always_inline
    @parameter
    fn butterfly_loop(idx: Int):
        zero_idx = 2*idx - idx%stride
        one_idx = zero_idx + stride

        if show:
            print("idx:", idx, ", pairs:", zero_idx, one_idx)


        var elem0_re = vector_re.data[zero_idx]
        var elem0_im = vector_im.data[zero_idx]
        var elem1_re = vector_re.data[one_idx]
        var elem1_im = vector_im.data[one_idx]

        elem0_orig_re = elem0_re
        elem0_orig_im = elem0_im

        elem1_orig_re = elem1_re
        elem1_orig_im = elem1_im

        elem0_re = elem0_orig_re.fma(gate_re[0][0], -gate_im[0][0]*elem0_orig_im + elem1_orig_re.fma(gate_re[0][1], -gate_im[0][1]*elem1_orig_im))
        elem0_im = elem0_orig_re.fma(gate_im[0][0], gate_re[0][0]*elem0_orig_im + elem1_orig_re.fma(gate_im[0][1], gate_re[0][1]*elem1_orig_im))
        elem1_re = elem0_orig_re.fma(gate_re[1][0], -gate_im[1][0]*elem0_orig_im + elem1_orig_re.fma(gate_re[1][1], -gate_im[1][1]*elem1_orig_im))
        elem1_im = elem0_orig_re.fma(gate_im[1][0], gate_re[1][0]*elem0_orig_im + elem1_orig_re.fma(gate_im[1][1], gate_re[1][1]*elem1_orig_im))

        vector_re.data[zero_idx] = elem0_re
        vector_im.data[zero_idx] = elem0_im
        vector_re.data[one_idx] = elem1_re
        vector_im.data[one_idx] = elem1_im

    @always_inline
    @parameter
    fn elementwise_fn[
        simd_width: Int, rank: Int, alignment: Int = 1
    ](idx: IndexList[rank]):
        butterfly_simd[simd_width](idx[0])

    @parameter
    @always_inline
    fn worker_loop(item_id: Int):
        start = item_id*chunk_size
        for idx in range(start, start + chunk_size):
            butterfly_loop(idx)


    @parameter
    @always_inline
    fn worker_simd(item_id: Int):
        var start = item_id*chunk_size
        for idx in range(start, start + chunk_size, simd_width):
            butterfly_simd[simd_width](idx)

#         @parameter
#         @always_inline
#         fn butterfly_simd_wrapper[width: Int](idx: Int):
#             butterfly_simd[simd_width](start + idx)
#
#         vectorize[butterfly_simd_wrapper, simd_width](chunk_size)

    try:
        if use_vectorize.isa[Bool]():
            if use_vectorize[Bool]:
                vectorize[butterfly_simd, simd_width](N//2)
            else:
                elementwise[elementwise_fn, simd_width, use_blocking_impl=False](N//2)
        elif use_vectorize[Int] == 0:
            for idx in range(N//2):
                butterfly_loop(idx)
        elif use_vectorize[Int] == 1:
            parallelize[worker_loop](num_work_items, num_threads)
        elif use_vectorize[Int] == 2:
            parallelize[worker_simd](num_work_items, num_threads)
        else:
            print("Unexpected use_vectorize parameter", use_vectorize[Int])
    except e:
        print("Caught an error:", e)

        #     print(vector)

    if show:
        for i in range(len(vector_re)):
            print(vector_re[i], vector_im[i], end=", ")
            #         print("\n")

fn test_loop[N:Int, gate: Gate, stride:Int]():
    var re = List[float_type](length=N, fill=0.0)
    re[0] = 1.0
    var im = List[float_type](length=N, fill=0.0)

    if stride > 0:
        transform[N](re, im, gate, stride)
    else:
        for i in range(Int(log2(Float32(N)))):
            transform[N](re, im, gate, 1 << i)

fn test_elementwise[N:Int, gate: Gate, stride: Int]():
    var re = List[float_type](length=N, fill=0.0)
    re[0] = 1.0
    var im = List[float_type](length=N, fill=0.0)

    if stride > 0:
        transform[N, False](re, im, gate, stride)
    else:
        for i in range(Int(log2(Float32(N)))):
            transform[N, False](re, im, gate, 1 << i)

fn test_vectorize[N:Int, gate: Gate, stride:Int]():
    var re = List[float_type](length=N, fill=0.0)
    re[0] = 1.0
    var im = List[float_type](length=N, fill=0.0)

    if stride > 0:
        transform[N, True](re, im, gate, stride)
    else:
        for i in range(Int(log2(Float32(N)))):
            transform[N, True](re, im, gate, 1 << i)

fn test_parallelize_loop[N:Int, gate: Gate, stride:Int]():
    var re = List[float_type](length=N, fill=0.0)
    re[0] = 1.0
    var im = List[float_type](length=N, fill=0.0)

    if stride > 0:
        transform[N, 1](re, im, gate, stride)
    else:
        for i in range(Int(log2(Float32(N)))):
            transform[N, 1](re, im, gate, 1 << i)

fn test_parallelize_simd[N:Int, gate: Gate, stride:Int]():
    var re = List[float_type](length=N, fill=0.0)
    re[0] = 1.0
    var im = List[float_type](length=N, fill=0.0)

    if stride > 0:
        transform[N, 2](re, im, gate, stride)
    else:
        for i in range(Int(log2(Float32(N)))):
            transform[N, 2](re, im, gate, 1 << i)

fn test_correctness[n: Int, stride: Int]() raises:
    alias N = 1 << n

    for gate in [X, H]:
        var re = List[float_type](length=N, fill=0.0)
        re[0] = 1.0
        var im = List[float_type](length=N, fill=0.0)

        for i in range(n):
            transform[N](re, im, gate, 1 << i)

        print("\n\nLoop")
        for i in range(min(8, len(re))):
            print(re[i], "+ i *", im[i], end=", ")

        var re1 = List[float_type](length=N, fill=0.0)
        re1[0] = 1.0
        var im1 = List[float_type](length=N, fill=0.0)

        for i in range(n):
            transform[N, False](re1, im1, gate, 1 << i)

        print("\n\nElementwise")
        for i in range(min(8, len(re))):
            print(re1[i], "+ i *", im1[i], end=", ")

        for i in range(N):
            assert_almost_equal(re[i], re1[i])
            assert_almost_equal(im[i], im1[i])

        var re2 = List[float_type](length=N, fill=0.0)
        re2[0] = 1.0
        var im2 = List[float_type](length=N, fill=0.0)


        for i in range(n):
            transform[N, True](re2, im2, gate, 1 << i)

        print("\n\nVectorize")
        for i in range(min(8, len(re))):
            print(re2[i], "+ i *", im2[i], end=", ")

        for i in range(N):
            assert_almost_equal(re[i], re2[i])
            assert_almost_equal(im[i], im2[i])

        var re3 = List[float_type](length=N, fill=0.0)
        re3[0] = 1.0
        var im3 = List[float_type](length=N, fill=0.0)

        for i in range(n):
            transform[N, 1](re3, im3, gate, 1 << i)

        print("\n\nParallel Loop")
        for i in range(min(8, len(re))):
            print(re3[i], "+ i *", im3[i], end=", ")

        for i in range(N):
            assert_almost_equal(re[i], re3[i])
            assert_almost_equal(im[i], im3[i])
        var re4 = List[float_type](length=N, fill=0.0)
        re4[0] = 1.0
        var im4 = List[float_type](length=N, fill=0.0)

        for i in range(n):
            transform[N, 2](re4, im4, gate, 1 << i)

        print("\n\nParallel SIMD")
        for i in range(min(8, len(re))):
            print(re4[i], "+ i *", im4[i], end=", ")

        for i in range(N):
            assert_almost_equal(re[i], re4[i])
            assert_almost_equal(im[i], im4[i])

def main():
    alias n = 25
    alias target = n//2
    alias stride = 0 # 1 << target

    alias N = 1 << n

    test_correctness[n, stride]()

    iters = 3

    print("\nGate H\n===================")

    print("n =", n, ", stride =", stride, ", iterations=", iters)

    t0 = benchmark.run[test_loop[N, H, stride]](2, iters).mean()
    print("loop", t0)

    t1 = benchmark.run[test_elementwise[N, H, stride]](2, iters).mean()
    print("elementwise", t1)
    print("speedup of elementwise over loop", t0/t1)

    t2 = benchmark.run[test_vectorize[N, H, stride]](2, iters).mean()
    print("vectorize", t2)
    print("speedup of vectorize over loop", t0/t2)

    t3 = benchmark.run[test_parallelize_loop[N, H, stride]](2, iters).mean()
    print("parallelize loop", t3)
    print("speedup of parallelize loop over loop", t0/t3)

    t4 = benchmark.run[test_parallelize_simd[N, H, stride]](2, iters).mean()
    print("parallelize loop", t4)
    print("speedup of parallelize simd over loop", t0/t4)


    print("\nGate X\n===================")

    print("n =", n, ", stride =", stride, ", iterations=", iters)
    t0 = benchmark.run[test_loop[N, X, stride]](2, iters).mean()
    print("loop", t0)

    t1 = benchmark.run[test_elementwise[N, X, stride]](2, iters).mean()
    print("elementwise", t1)
    print("speedup of elementwise over loop", t0/t1)

    t2 = benchmark.run[test_vectorize[N, X, stride]](2, iters).mean()
    print("vectorize", t2)
    print("speedup of vectorize over loop", t0/t2)

    t3 = benchmark.run[test_parallelize_loop[N, X, stride]](2, iters).mean()
    print("parallelize loop", t3)
    print("speedup of parallelize loop over loop", t0/t3)

    t4 = benchmark.run[test_parallelize_simd[N, X, stride]](2, iters).mean()
    print("parallelize loop", t4)
    print("speedup of parallelize simd over loop", t0/t4)

# Gate H
# ===================
# n = 25 , stride = 0 , iterations= 3
# loop 0.8700783333333333
# elementwise 0.07059766666666667
# speedup of elementwise over loop 12.324463037021996
# vectorize 0.279345
# speedup of vectorize over loop 3.1147088128777436
# parallelize loop 0.234823
# speedup of parallelize loop over loop 3.705251756997114
# parallelize loop 0.081918
# speedup of parallelize simd over loop 10.621332714828648
#
# Gate X
# ===================
# n = 25 , stride = 0 , iterations= 3
# loop 0.8694423333333333
# elementwise 0.07165333333333333
# speedup of elementwise over loop 12.134010978786751
# vectorize 0.27941699999999997
# speedup of vectorize over loop 3.11163004875628
# parallelize loop 0.24091333333333334
# speedup of parallelize loop over loop 3.6089423582477793
# parallelize loop 0.081637
# speedup of parallelize simd over loop 10.650101465430298

# Architecture:             x86_64
# CPU op-mode(s):         32-bit, 64-bit
# Address sizes:          46 bits physical, 57 bits virtual
# Byte Order:             Little Endian
# CPU(s):                   30
# On-line CPU(s) list:    0-29
# Vendor ID:                GenuineIntel
# Model name:             Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz
# CPU family:           6
# Model:                106
# Thread(s) per core:   1
# Core(s) per socket:   1
# Socket(s):            30
# Stepping:             6
# BogoMIPS:             5187.93
# Flags:                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss syscall nx pdpe1gb rdtsc
# p lm constant_tsc arch_perfmon rep_good nopl xtopology cpuid tsc_known_freq pni pclmulqdq vmx ssse3 fma cx16 pdcm pcid sse4_1
# sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch cpuid_fault ssb
# d ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid
# avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves
# wbnoinvd arat vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la5
# 7 rdpid fsrm md_clear arch_capabilities
# Virtualization features:
# Virtualization:         VT-x
# Hypervisor vendor:      KVM
# Virtualization type:    full
# Caches (sum of all):
# L1d:                    960 KiB (30 instances)
# L1i:                    960 KiB (30 instances)
# L2:                     120 MiB (30 instances)
# L3:                     480 MiB (30 instances)
# NUMA:
# NUMA node(s):           1
# NUMA node0 CPU(s):      0-29
# Vulnerabilities:
# Gather data sampling:   Unknown: Dependent on hypervisor status
# Itlb multihit:          Not affected
# L1tf:                   Not affected
# Mds:                    Not affected
# Meltdown:               Not affected
# Mmio stale data:        Vulnerable: Clear CPU buffers attempted, no microcode; SMT Host state unknown
# Reg file data sampling: Not affected
# Retbleed:               Not affected
# Spec rstack overflow:   Not affected
# Spec store bypass:      Mitigation; Speculative Store Bypass disabled via prctl
# Spectre v1:             Mitigation; usercopy/swapgs barriers and __user pointer sanitization
# Spectre v2:             Mitigation; Enhanced / Automatic IBRS; IBPB conditional; RSB filling; PBRSB-eIBRS SW sequence; BHI SW loop, KVM SW loop
# Srbds:                  Not affected
# Tsx async abort:        Mitigation; TSX disabled

# Gate H
# ===================
# n = 27 , stride = 0 , iterations= 3
# loop 8.522051395666667
# elementwise 0.4810115156666666
# speedup of elementwise over loop 17.716938406049128
# vectorize 2.0300095823333333
# speedup of vectorize over loop 4.198035058470636
# parallelize loop 1.418304324
# speedup of parallelize loop over loop 6.008619766195303
# parallelize loop 0.5622232976666667
# speedup of parallelize simd over loop 15.15776993062151
#
# Gate X
# ===================
# n = 27 , stride = 0 , iterations= 3
# loop 8.524472887333333
# elementwise 0.47970799866666664
# speedup of elementwise over loop 17.77012872628107
# vectorize 2.028346640666667
# speedup of vectorize over loop 4.202670646340584
# parallelize loop 1.4226437433333334
# speedup of parallelize loop over loop 5.991994079529721
# parallelize loop 0.5617238403333333
# speedup of parallelize simd over loop 15.175558299741763