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
#     state = List[ComplexSIMD[dtype, 2*simd_width]](capacity=N//2//simd_width)


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

#         sq2_simd = sqrt(0.5).cast[dtype]()

        elem0_orig_re = elem0_re
        elem0_orig_im = elem0_im

        elem1_orig_re = elem1_re
        elem1_orig_im = elem1_im

#         z0*g00 + z1*g01: z0.re.fma(g00.re, -z0.im * g00.im) + z1.re.fma(g01.re, -z1.im * g01.im)
#         z0.re.fma(gate[0][0].re, -gate[0][0].im*z0.im + z1.re.fma(gate[0][1].re, -gate[0][1].im*z1.im)),

#         z0*g00 + z1*g01: z0.re.fma(g00.im, z0.im * g00.re) + z1.re.fma(g01.im, z1.im * g01.re)
#         z0.re.fma(gate[0][0].im, gate[0][0].im*z0.re + z1.re.fma(gate[0][1].im, gate[0][1].im*z1.re))

#         z0*g10 + z1*g11: z0.re.fma(g10.re, -z0.im * g10.im) + z1.re.fma(g11.re, -z1.im * g11.im)
#         z.re.fma(gate[1][0].re, -gate[1][0].im*z0.im + z1.re.fma(gate[1][1].re, -gate[1][1].im*z1.im)),

#         z0*g10 + z1*g11: z0.re.fma(g10.im, z0.im * g10.re) + z1.re.fma(g11.im, z1.im * g11.re)
#         z.re.fma(gate[1][0].im, gate[1][0].im*z0.re + z1.re.fma(gate[1][1].im, gate[1][1].im*z1.re))

#     z0.re.fma(z1.re, -z1.im * z0.im),
#     z0.re.fma(z1.im, z1.re * z0.im),

        elem0_re = elem0_orig_re.fma(gate_re[0][0], -gate_im[0][0]*elem0_orig_im + elem1_orig_re.fma(gate_re[0][1], -gate_im[0][1]*elem1_orig_im))
        elem0_im = elem0_orig_re.fma(gate_im[0][0], gate_re[0][0]*elem0_orig_im + elem1_orig_re.fma(gate_im[0][1], gate_re[0][1]*elem1_orig_im))
        elem1_re = elem0_orig_re.fma(gate_re[1][0], -gate_im[1][0]*elem0_orig_im + elem1_orig_re.fma(gate_re[1][1], -gate_im[1][1]*elem1_orig_im))
        elem1_im = elem0_orig_re.fma(gate_im[1][0], gate_re[1][0]*elem0_orig_im + elem1_orig_re.fma(gate_im[1][1], gate_re[1][1]*elem1_orig_im))

#         elem0_re = elem0_orig_re.fma(0, -0*elem0_orig_im + elem1_orig_re.fma(1, -0*elem1_orig_im))
#         elem0_im = elem0_orig_re.fma(0, gate_re[0][0]*elem0_orig_im + elem1_orig_re.fma(0, 1*elem1_orig_im))
#         elem1_re = elem0_orig_re.fma(1, -0*elem0_orig_im + elem1_orig_re.fma(0, -0*elem1_orig_im))
#         elem1_im = elem0_orig_re.fma(0, 1*elem0_orig_im + elem1_orig_re.fma(0, 1*elem1_orig_im))

#         elem0_re = elem1_orig_re
#         elem0_im = elem1_orig_im
#         elem1_re = elem0_orig_re
#         elem1_im = elem0_orig_im

#         res0_re = (elem0_orig_re + elem1_re)*sq2_simd
#         res0_im = (elem0_im + elem1_im)*sq2_simd
#         res1_re = (elem0_orig_re - elem1_re)*sq2_simd
#         res1_im = (elem0_im - elem1_im)*sq2_simd

#         if show:
#             print("loaded SIMDs:", elem0_re, elem0_im, elem1_re, elem1_im, "->", res0_re, res0_im, res1_re, res1_im)

        vector_re.store[width=simd_width](zero_idx, elem0_re)
        vector_im.store[width=simd_width](zero_idx, elem0_im)
        vector_re.store[width=simd_width](one_idx, elem1_re)
        vector_im.store[width=simd_width](one_idx, elem1_im)

    fn butterfly_loop(idx: Int):
        zero_idx = 2*idx - idx%stride
        one_idx = zero_idx + stride

        if show:
            print("idx:", idx, ", pairs:", zero_idx, one_idx)

#         var elem0 = Amplitude(vector_re.data[zero_idx], vector_im.data[zero_idx])
#         var elem1 = Amplitude(vector_re.data[one_idx], vector_im.data[one_idx])
# #         s = (elem0 + elem1)*sq2
# #         d = (elem0 - elem1)*sq2
#         s = elem1
#         d = elem0
#         vector_re.data[zero_idx] = s.re
#         vector_im.data[zero_idx] = s.im
#         vector_re.data[one_idx] = d.re
#         vector_im.data[one_idx] = d.im

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
    fn worker(thread_id: Int):
        start = thread_id*chunk_size
        for idx in range(start, start + chunk_size):
            butterfly_loop(idx)

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
            parallelize[worker](num_work_items, num_threads)
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

fn test_parallelize[N:Int, gate_re: InlineArray[InlineArray[float_type, 2], 2], gate_im: InlineArray[InlineArray[float_type, 2], 2], stride:Int]():
    var re = List[float_type](length=N, fill=0.0)
    re[0] = 1.0
    var im = List[float_type](length=N, fill=0.0)

    if stride > 0:
        transform[N, 1](re, im, gate_re, gate_im, stride)
    else:
        for stride in range(Int(log2(Float32(N)))):
            transform[N, 1](re, im, gate_re, gate_im, stride)

fn test_correctness[n: Int, stride: Int]() raises:
    alias N = 1 << n

    for (gate_re, gate_im) in [(X_re, X_im), (H_re, H_im)]:
        var re = List[float_type](length=N, fill=0.0)
        re[0] = 1.0
        var im = List[float_type](length=N, fill=0.0)

        for i in range(n):
            transform[N](re, im, gate_re, gate_im, 1 << i)

        for i in range(min(8, len(re))):
            print(re[i], "+ i *", im[i], end=", ")
        print("\n")

        var re1 = List[float_type](length=N, fill=0.0)
        re1[0] = 1.0
        var im1 = List[float_type](length=N, fill=0.0)


        for i in range(n):
            transform[N, False](re1, im1, gate_re, gate_im, 1 << i)

        for i in range(min(8, len(re))):
            print(re1[i], "+ i *", im1[i], end=", ")
        print("\n")

        for i in range(N):
            assert_almost_equal(re[i], re1[i])
            assert_almost_equal(im[i], im1[i])

        var re2 = List[float_type](length=N, fill=0.0)
        re2[0] = 1.0
        var im2 = List[float_type](length=N, fill=0.0)


        for i in range(n):
            transform[N, True](re2, im2, gate_re, gate_im, 1 << i)

        for i in range(min(8, len(re))):
            print(re2[i], "+ i *", im2[i], end=", ")
        print("\n")

        for i in range(N):
            assert_almost_equal(re[i], re2[i])
            assert_almost_equal(im[i], im2[i])

        var re3 = List[float_type](length=N, fill=0.0)
        re3[0] = 1.0
        var im3 = List[float_type](length=N, fill=0.0)

        for i in range(n):
            transform[N, 1](re3, im3, gate_re, gate_im, 1 << i)

        for i in range(min(8, len(re))):
            print(re3[i], "+ i *", im3[i], end=", ")
        print("\n")

        for i in range(N):
            assert_almost_equal(re[i], re3[i])
            assert_almost_equal(im[i], im3[i])

def main():
    alias n = 20
#     alias target = n-1
    alias stride = 0 # 1 << target

    alias N = 1 << n

    test_correctness[n, stride]()

    iters = 3

    print("Gate H\n===================")

    print("n =", n, ", stride =", stride, ", iterations=", iters)
    t0 = benchmark.run[test_loop[N, H_re, H_im, stride]](2, iters).mean()
    print("loop", t0)

    t1 = benchmark.run[test_elementwise[N, H_re, H_im, stride]](2, iters).mean()
    print("elementwise", t1)
    print("speedup of elementwise over loop", t0/t1)

    t2 = benchmark.run[test_vectorize[N, H_re, H_im, stride]](2, iters).mean()
    print("vectorize", t2)
    print("speedup of vectorize over loop", t0/t2)

    t3 = benchmark.run[test_parallelize[N, H_re, H_im, stride]](2, iters).mean()
    print("parallelize loop", t3)
    print("speedup of parallelize loop over loop", t0/t3)

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

    t3 = benchmark.run[test_parallelize[N, X_re, X_im, stride]](2, iters).mean()
    print("parallelize loop", t3)
    print("speedup of parallelize loop over loop", t0/t3)


# Gate H
# ===================
# n = 18 , stride = 0 , iterations= 10
# loop 0.0036177
# vectorize 0.0011246000000000001
# speedup of vectorize over loop 3.216877111861995
#
# Gate X
# ===================
# n = 18 , stride = 0 , iterations= 10
# loop 0.0032176999999999996
# vectorize 0.001125
# speedup of vectorize over loop 2.8601777777777775