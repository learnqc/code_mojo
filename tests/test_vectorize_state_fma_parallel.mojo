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


# alias sq2_float = sqrt(0.5).cast[dtype]()
alias H: InlineArray[InlineArray[Amplitude, 2], 2] = [[sq2, sq2], [sq2, -sq2]]
alias H_re: InlineArray[InlineArray[float_type, 2], 2] =  [[H[0][0].re, H[0][1].re], [H[1][0].re, H[1][1].re]]
alias H_im: InlineArray[InlineArray[float_type, 2], 2]  = [[H[0][0].im, H[0][1].im], [H[1][0].im, H[1][1].im]]

alias X: InlineArray[InlineArray[Amplitude, 2], 2] = [[Amplitude(sqrt(0.0).cast[dtype](), 0.0), Amplitude(sqrt(1.0).cast[dtype](), 0)],
    [Amplitude(sqrt(1.0).cast[dtype](), 0.0), Amplitude(sqrt(0.0).cast[dtype](), 0)]]
alias X_re: InlineArray[InlineArray[float_type, 2], 2] =  [[X[0][0].re, X[0][1].re], [X[1][0].re, X[1][1].re]]
alias X_im: InlineArray[InlineArray[float_type, 2], 2]  = [[X[0][0].im, X[0][1].im], [X[1][0].im, X[1][1].im]]


fn transform[N: Int, use_vectorize: simd_type = 0, show: Bool=False](mut re: List[float_type], mut im: List[float_type],
    gate_re: InlineArray[InlineArray[float_type, 2], 2], gate_im: InlineArray[InlineArray[float_type, 2], 2], stride: Int) :

    alias num_work_items = 8
    alias num_threads = num_work_items
    alias chunk_size = N//2//num_work_items

    var vector_re = NDBuffer[dtype, 1, _, N](re)
    var vector_im = NDBuffer[dtype, 1, _, N](im)


    @always_inline
    #     @__copy_capture(vector_re, vector_im)
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
    fn worker_loop(thread_id: Int):
        start = thread_id*chunk_size
        for idx in range(start, start + chunk_size):
            butterfly_loop(idx)

    @parameter
    @always_inline
    fn worker_simd(thread_id: Int):
        start = thread_id*chunk_size
        for idx in range(start, start + chunk_size, simd_width):
            butterfly_simd[simd_width](idx)

    try:
        if use_vectorize.isa[Bool]():
            if use_vectorize[Bool]:
                vectorize[butterfly_simd, simd_width](N//2)
            else:
                elementwise[elementwise_fn, simd_width](N//2)
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

fn test_loop[N:Int, gate_re: InlineArray[InlineArray[float_type, 2], 2], gate_im: InlineArray[InlineArray[float_type, 2], 2], stride:Int]():
    var re = List[float_type](length=N, fill=0.0)
    re[0] = 1.0
    var im = List[float_type](length=N, fill=0.0)

    if stride > 0:
        transform[N](re, im, gate_re, gate_im, stride)
    else:
        for stride in range(Int(log2(Float32(N)))):
            transform[N](re, im, gate_re, gate_im, stride)

fn test_elementwise[N:Int, gate_re: InlineArray[InlineArray[float_type, 2], 2], gate_im: InlineArray[InlineArray[float_type, 2], 2], stride: Int]():
    var re = List[float_type](length=N, fill=0.0)
    re[0] = 1.0
    var im = List[float_type](length=N, fill=0.0)

    if stride > 0:
        transform[N, False](re, im, gate_re, gate_im, stride)
    else:
        for stride in range(Int(log2(Float32(N)))):
            transform[N, False](re, im, gate_re, gate_im, stride)

fn test_vectorize[N:Int, gate_re: InlineArray[InlineArray[float_type, 2], 2], gate_im: InlineArray[InlineArray[float_type, 2], 2], stride:Int]():
    var re = List[float_type](length=N, fill=0.0)
    re[0] = 1.0
    var im = List[float_type](length=N, fill=0.0)

    if stride > 0:
        transform[N, True](re, im, gate_re, gate_im, stride)
    else:
        for stride in range(Int(log2(Float32(N)))):
            transform[N, True](re, im, gate_re, gate_im, stride)

fn test_parallelize_loop[N:Int, gate_re: InlineArray[InlineArray[float_type, 2], 2], gate_im: InlineArray[InlineArray[float_type, 2], 2], stride:Int]():
    var re = List[float_type](length=N, fill=0.0)
    re[0] = 1.0
    var im = List[float_type](length=N, fill=0.0)

    if stride > 0:
        transform[N, 1](re, im, gate_re, gate_im, stride)
    else:
        for stride in range(Int(log2(Float32(N)))):
            transform[N, 1](re, im, gate_re, gate_im, stride)

fn test_parallelize_simd[N:Int, gate_re: InlineArray[InlineArray[float_type, 2], 2], gate_im: InlineArray[InlineArray[float_type, 2], 2], stride:Int]():
    var re = List[float_type](length=N, fill=0.0)
    re[0] = 1.0
    var im = List[float_type](length=N, fill=0.0)

    if stride > 0:
        transform[N, 2](re, im, gate_re, gate_im, stride)
    else:
        for stride in range(Int(log2(Float32(N)))):
            transform[N, 2](re, im, gate_re, gate_im, stride)

fn test_correctness[n: Int, stride: Int]() raises:
    alias N = 1 << n

    for (gate_re, gate_im) in [(X_re, X_im), (H_re, H_im)]:
        var re = List[float_type](length=N, fill=0.0)
        re[0] = 1.0
        var im = List[float_type](length=N, fill=0.0)

        for i in range(n):
            transform[N](re, im, gate_re, gate_im, 1 << i)

        print("\n\nLoop")
        for i in range(min(8, len(re))):
            print(re[i], "+ i *", im[i], end=", ")

        var re1 = List[float_type](length=N, fill=0.0)
        re1[0] = 1.0
        var im1 = List[float_type](length=N, fill=0.0)

        for i in range(n):
            transform[N, False](re1, im1, gate_re, gate_im, 1 << i)

        print("\n\nElemntwise")
        for i in range(min(8, len(re))):
            print(re1[i], "+ i *", im1[i], end=", ")

        for i in range(N):
            assert_almost_equal(re[i], re1[i])
            assert_almost_equal(im[i], im1[i])

        var re2 = List[float_type](length=N, fill=0.0)
        re2[0] = 1.0
        var im2 = List[float_type](length=N, fill=0.0)


        for i in range(n):
            transform[N, True](re2, im2, gate_re, gate_im, 1 << i)

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
            transform[N, 1](re3, im3, gate_re, gate_im, 1 << i)

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
            transform[N, 2](re4, im4, gate_re, gate_im, 1 << i)

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
    t0 = benchmark.run[test_loop[N, H_re, H_im, stride]](2, iters).mean()
    print("loop", t0)

    t1 = benchmark.run[test_elementwise[N, H_re, H_im, stride]](2, iters).mean()
    print("elementwise", t1)
    print("speedup of elementwise over loop", t0/t1)

    t2 = benchmark.run[test_vectorize[N, H_re, H_im, stride]](2, iters).mean()
    print("vectorize", t2)
    print("speedup of vectorize over loop", t0/t2)

    t3 = benchmark.run[test_parallelize_loop[N, H_re, H_im, stride]](2, iters).mean()
    print("parallelize loop", t3)
    print("speedup of parallelize loop over loop", t0/t3)

    t4 = benchmark.run[test_parallelize_simd[N, H_re, H_im, stride]](2, iters).mean()
    print("parallelize loop", t4)
    print("speedup of parallelize simd over loop", t0/t4)


    print("\nGate X\n===================")

    print("n =", n, ", stride =", stride, ", iterations=", iters)
    t0 = benchmark.run[test_loop[N, X_re, X_im, stride]](2, iters).mean()
    print("loop", t0)

    t1 = benchmark.run[test_elementwise[N, X_re, X_im, stride]](2, iters).mean()
    print("elementwise", t1)
    print("speedup of elementwise over loop", t0/t1)

    t2 = benchmark.run[test_vectorize[N, X_re, X_im, stride]](2, iters).mean()
    print("vectorize", t2)
    print("speedup of vectorize over loop", t0/t2)

    t3 = benchmark.run[test_parallelize_loop[N, X_re, X_im, stride]](2, iters).mean()
    print("parallelize loop", t3)
    print("speedup of parallelize loop over loop", t0/t3)

    t4 = benchmark.run[test_parallelize_simd[N, X_re, X_im, stride]](2, iters).mean()
    print("parallelize loop", t4)
    print("speedup of parallelize simd over loop", t0/t4)

# Gate H
# ===================
# n = 25 , stride = 0 , iterations= 3
# loop 0.6555483333333333
# elementwise 0.06367666666666667
# speedup of elementwise over loop 10.294953672198082
# vectorize 0.210143
# speedup of vectorize over loop 3.1195344757300187
# parallelize loop 0.207108
# speedup of parallelize loop over loop 3.165248726912207
# parallelize loop 0.073976
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