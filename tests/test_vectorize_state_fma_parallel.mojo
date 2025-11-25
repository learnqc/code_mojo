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
        for stride in range(Int(log2(Float32(N)))):
            transform[N](re, im, gate, stride)

fn test_elementwise[N:Int, gate: Gate, stride: Int]():
    var re = List[float_type](length=N, fill=0.0)
    re[0] = 1.0
    var im = List[float_type](length=N, fill=0.0)

    if stride > 0:
        transform[N, False](re, im, gate, stride)
    else:
        for stride in range(Int(log2(Float32(N)))):
            transform[N, False](re, im, gate, stride)

fn test_vectorize[N:Int, gate: Gate, stride:Int]():
    var re = List[float_type](length=N, fill=0.0)
    re[0] = 1.0
    var im = List[float_type](length=N, fill=0.0)

    if stride > 0:
        transform[N, True](re, im, gate, stride)
    else:
        for stride in range(Int(log2(Float32(N)))):
            transform[N, True](re, im, gate, stride)

fn test_parallelize_loop[N:Int, gate: Gate, stride:Int]():
    var re = List[float_type](length=N, fill=0.0)
    re[0] = 1.0
    var im = List[float_type](length=N, fill=0.0)

    if stride > 0:
        transform[N, 1](re, im, gate, stride)
    else:
        for stride in range(Int(log2(Float32(N)))):
            transform[N, 1](re, im, gate, stride)

fn test_parallelize_simd[N:Int, gate: Gate, stride:Int]():
    var re = List[float_type](length=N, fill=0.0)
    re[0] = 1.0
    var im = List[float_type](length=N, fill=0.0)

    if stride > 0:
        transform[N, 2](re, im, gate, stride)
    else:
        for stride in range(Int(log2(Float32(N)))):
            transform[N, 2](re, im, gate, stride)

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

#         var re1 = List[float_type](length=N, fill=0.0)
#         re1[0] = 1.0
#         var im1 = List[float_type](length=N, fill=0.0)
#
#         for i in range(n):
#             transform[N, False](re1, im1, gate, 1 << i)
#
#         print("\n\nElemntwise")
#         for i in range(min(8, len(re))):
#             print(re1[i], "+ i *", im1[i], end=", ")
#
#         for i in range(N):
#             assert_almost_equal(re[i], re1[i])
#             assert_almost_equal(im[i], im1[i])

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

#     t1 = benchmark.run[test_elementwise[N, H, stride]](2, iters).mean()
#     print("elementwise", t1)
#     print("speedup of elementwise over loop", t0/t1)

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
# loop 0.6555483333333333
# elementwise 0.06367666666666667
# speedup of elementwise over loop     10.294953672198082
# speedup of parallelize simd over loop 8.86163530514401
#
# Gate X
# ===================
# n = 25 , stride = 0 , iterations= 3
# loop 0.579932
# elementwise 0.06832133333333333
# speedup of elementwise over loop 8.488300384457759
# vectorize 0.20959133333333335
# speedup of vectorize over loop 2.7669655551915313
# parallelize loop 0.20850766666666667
# speedup of parallelize loop over loop 2.781346169525341
# parallelize loop 0.07186133333333333
# speedup of parallelize simd over loop 8.070153629211816

# Gate H
# ===================
# n = 25 , stride = 0 , iterations= 3
# loop 0.836788
# elementwise 0.07062366666666667
# speedup of elementwise over loop 11.84854935314413
# vectorize 0.27609500000000003
# speedup of vectorize over loop 3.030797370470309
# parallelize loop 0.2305596666666667
# speedup of parallelize loop over loop 3.629377211105151
# parallelize loop 0.07998433333333334
# speedup of parallelize simd over loop 10.46189878851275
#
# Gate X
# ===================
# n = 25 , stride = 0 , iterations= 3
# loop 0.8385503333333334
# elementwise 0.076264
# speedup of elementwise over loop 10.995362600090914
# vectorize 0.2754543333333333
# speedup of vectorize over loop 3.0442444785161005
# parallelize loop 0.230056
# speedup of parallelize loop over loop 3.644983540239478
# parallelize loop 0.08002766666666666
# speedup of parallelize simd over loop 10.47825543666149

# Gate H
# ===================
# n = 30 , stride = 0 , iterations= 3
# loop 33.008257666666665
# elementwise 2.925299666666667
# speedup of elementwise over loop 11.283718397397234
# vectorize 10.932221333333333
# speedup of vectorize over loop 3.0193550478182782
# parallelize loop 9.398081333333334
# speedup of parallelize loop over loop 3.5122336672691064
# parallelize loop 3.582056
# speedup of parallelize simd over loop 9.214891578095559
#
# Gate X
# ===================
# n = 30 , stride = 0 , iterations= 3
# loop 32.950286999999996
# elementwise 2.9463000000000004
# speedup of elementwise over loop 11.183615721413295
# vectorize 10.933810000000001
# speedup of vectorize over loop 3.0136143759586083
# parallelize loop 9.333709666666666
# speedup of parallelize loop over loop 3.530245548313427
# parallelize loop 3.565137
# speedup of parallelize simd over loop 9.242362074725317


# Lambda gpu_1x_a100_sxm4
# Architecture:             x86_64
# CPU op-mode(s):         32-bit, 64-bit
# Address sizes:          48 bits physical, 48 bits virtual
# Byte Order:             Little Endian
# CPU(s):                   30
# On-line CPU(s) list:    0-29
# Vendor ID:                AuthenticAMD
# Model name:             AMD EPYC 7J13 64-Core Processor
# CPU family:           25
# Model:                1
# Thread(s) per core:   1
# Core(s) per socket:   1
# Socket(s):            30
# Stepping:             1
# BogoMIPS:             4899.99
# Flags:                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 syscall nx mmxext fxsr_opt
# pdpe1gb rdtscp lm rep_good nopl cpuid extd_apicid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe
# popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy svm cr8_legacy abm sse4a misalignsse 3dnow
# prefetch osvw perfctr_core ssbd ibrs ibpb stibp vmmcall fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid rdseed adx smap
# clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 xsaves clzero xsaveerptr wbnoinvd arat npt nrip_save umip pku ospke vaes vpclm
# ulqdq rdpid fsrm arch_capabilities
# Virtualization features:
# Virtualization:         AMD-V
# Hypervisor vendor:      KVM
# Virtualization type:    full
# Caches (sum of all):
# L1d:                    1.9 MiB (30 instances)
# L1i:                    1.9 MiB (30 instances)
# L2:                     15 MiB (30 instances)
# L3:                     480 MiB (30 instances)

# 8 threads

# Gate H
# ===================
# n = 25 , stride = 0 , iterations= 3
# loop 2.09853828
# elementwise 0.07497758066666667
# speedup of elementwise over loop 27.98887695949574
# vectorize 0.3591315256666667
# speedup of vectorize over loop 5.843369712821564
# parallelize loop 0.955386411
# speedup of parallelize loop over loop 2.1965335238581285
# parallelize loop 0.16315704
# speedup of parallelize simd over loop 12.862076193586253
#
# Gate X
# ===================
# n = 25 , stride = 0 , iterations= 3
# loop 2.0940641793333334
# elementwise 0.08920866833333334
# speedup of elementwise over loop 23.473774672980678
# vectorize 0.3520875273333333
# speedup of vectorize over loop 5.947567058661556
# parallelize loop 0.9531740193333333
# speedup of parallelize loop over loop 2.1969379534683067
# parallelize loop 0.16206652600000002
# speedup of parallelize simd over loop 12.921016023588567

# 16 threads
#
# Gate H
# ===================
# n = 25 , stride = 0 , iterations= 3
# loop 2.0972429133333335
# elementwise 0.075580493
# speedup of elementwise over loop 27.74846828973891
# vectorize 0.36136831100000005
# speedup of vectorize over loop 5.803616004761782
# parallelize loop 0.48699397299999997
# speedup of parallelize loop over loop 4.306506917146865
# parallelize loop 0.12527313933333334
# speedup of parallelize simd over loop 16.741361512086637

# 32 threads
#
# Gate H
# ===================
# n = 25 , stride = 0 , iterations= 3
# loop 2.0991119443333335
# elementwise 0.07654387233333333
# speedup of elementwise over loop 27.423644510590197
# vectorize 0.36699143733333334
# speedup of vectorize over loop 5.719784525726519
# parallelize loop 0.48927322166666665
# speedup of parallelize loop over loop 4.290265339237025
# parallelize loop 0.09341269299999999
# speedup of parallelize simd over loop 22.471378106327943
#
# Gate X
# ===================
# n = 25 , stride = 0 , iterations= 3
# loop 2.1050625513333334
# elementwise 0.075834143
# speedup of elementwise over loop 27.758770232734523
# vectorize 0.3855356126666667
# speedup of vectorize over loop 5.460098839567815
# parallelize loop 0.49268570066666667
# speedup of parallelize loop over loop 4.2726276579265745
# parallelize loop 0.09257263333333333
# speedup of parallelize simd over loop 22.73957729768229
#
# Gate X
# ===================
# n = 25 , stride = 0 , iterations= 3
# loop 2.0969899696666667
# elementwise 0.09358710766666667
# speedup of elementwise over loop 22.406825276998713
# vectorize 0.35252179266666667
# speedup of vectorize over loop 5.948539957782166
# parallelize loop 0.4957362646666667
# speedup of parallelize loop over loop 4.230051580101133
# parallelize loop 0.12414369200000001
# speedup of parallelize simd over loop 16.89163529683543

# blocking elemntwise
# Gate H
# ===================
# n = 25 , stride = 0 , iterations= 3
# loop 2.098931217666667
# elementwise 0.3651035976666666
# speedup of elementwise over loop 5.748864790926972
# vectorize 0.358874994
# speedup of vectorize over loop 5.8486415959833264
# parallelize loop 0.4808819673333333
# speedup of parallelize loop over loop 4.364753432751927
# parallelize loop 0.09509566800000001
# speedup of parallelize simd over loop 22.071785832207063
#
# Gate X
# ===================
# n = 25 , stride = 0 , iterations= 3
# loop 2.097415236666667
# elementwise 0.3653520476666667
# speedup of elementwise over loop 5.740806025481124
# vectorize 0.358897378
# speedup of vectorize over loop 5.844052827453833
# parallelize loop 0.48184765999999996
# speedup of parallelize loop over loop 4.352859650011929
# parallelize loop 0.09394677
# speedup of parallelize simd over loop 22.32557049770489