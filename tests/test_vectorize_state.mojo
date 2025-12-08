from math import sqrt, log2
from sys.info import simd_width_of

from complex import ComplexSIMD

from algorithm import elementwise, vectorize
from buffer import NDBuffer
from testing import assert_almost_equal

from utils.index import IndexList
from utils.variant import Variant

import benchmark

alias dtype = DType.float32
alias float_type = Scalar[dtype]

alias Amplitude = ComplexSIMD[dtype, 1]
alias sq_half: Amplitude = Amplitude(sqrt(0.5).cast[dtype](), 0)

alias simd_width = simd_width_of[dtype]()

alias simd_type = Variant[Int, Bool]

fn transform[N: Int, use_vectorize: simd_type = 0, show: Bool=False](mut re: List[float_type], mut im: List[float_type], stride: Int) :

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

        sq_half_simd = sqrt(0.5).cast[dtype]()
        res0_re = (elem0_re + elem1_re)*sq_half_simd
        res0_im = (elem0_im + elem1_im)*sq_half_simd
        res1_re = (elem0_re - elem1_re)*sq_half_simd
        res1_im = (elem0_im - elem1_im)*sq_half_simd

        if show:
            print("loaded SIMDs:", elem0_re, elem0_im, elem1_re, elem1_im, "->", res0_re, res0_im, res1_re, res1_im)

        vector_re.store[width=simd_width](zero_idx, res0_re)
        vector_im.store[width=simd_width](zero_idx, res0_im)
        vector_re.store[width=simd_width](one_idx, res1_re)
        vector_im.store[width=simd_width](one_idx, res1_im)

    fn butterfly_loop(idx: Int):
        zero_idx = 2*idx - idx%stride
        one_idx = zero_idx + stride

        if show:
            print("idx:", idx, ", pairs:", zero_idx, one_idx)

        var elem0 = Amplitude(vector_re.data[zero_idx], vector_im.data[zero_idx])
        var elem1 = Amplitude(vector_re.data[one_idx], vector_im.data[one_idx])
        s = (elem0 + elem1)*sq_half
        d = (elem0 - elem1)*sq_half
        vector_re.data[zero_idx] = s.re
        vector_im.data[zero_idx] = s.im
        vector_re.data[one_idx] = d.re
        vector_im.data[one_idx] = d.im

    @always_inline
    @parameter
    fn elementwise_fn[
        simd_width: Int, rank: Int, alignment: Int = 1
    ](idx: IndexList[rank]):
        butterfly_simd[simd_width](idx[0])

    try:
        if use_vectorize.isa[Bool]():
            if use_vectorize[Bool]:
                vectorize[butterfly_simd, simd_width](N//2)
            else:
                elementwise[elementwise_fn, simd_width](N//2)
        else:
            for idx in range(N//2):
                butterfly_loop(idx)
    except e:
        print("Caught an error:", e)

#     print(vector)

    if show:
        for i in range(len(vector_re)):
            print(vector_re[i], vector_im[i], end=", ")
#         print("\n")

fn test_loop[N:Int, stride:Int]():
    var re = List[float_type](length=N, fill=0.0)
    re[0] = 1.0
    var im = List[float_type](length=N, fill=0.0)

    if stride > 0:
        transform[N](re, im, stride)
    else:
        for stride in range(Int(log2(Float32(N)))):
            transform[N](re, im, stride)

fn test_elementwise[N:Int, stride: Int]():
    var re = List[float_type](length=N, fill=0.0)
    re[0] = 1.0
    var im = List[float_type](length=N, fill=0.0)

    if stride > 0:
        transform[N, False](re, im, stride)
    else:
        for stride in range(Int(log2(Float32(N)))):
            transform[N, False](re, im, stride)

fn test_vectorize[N:Int, stride:Int]():
    var re = List[float_type](length=N, fill=0.0)
    re[0] = 1.0
    var im = List[float_type](length=N, fill=0.0)

    if stride > 0:
        transform[N, True](re, im, stride)
    else:
        for stride in range(Int(log2(Float32(N)))):
            transform[N, True](re, im, stride)

def main():
    alias n = 16
    alias target = n-1
    alias stride = 0 # 1 << target

    alias N = 1 << n

    #     var ptr = UnsafePointer[float_type].alloc(N)
    #     var ptr = InlineArray[float_type, N](uninitialized=True)

    var re = List[float_type](length=N, fill=0.0)
    re[0] = 1.0
    var im = List[float_type](length=N, fill=0.0)

    for i in range(n):
        transform[N](re, im, 1 << i)

#     for i in range(len(re)):
#         print(re[i], "+ i *", im[i], end=", ")
#     print("\n")

    var re1 = List[float_type](length=N, fill=0.0)
    re1[0] = 1.0
    var im1 = List[float_type](length=N, fill=0.0)


    for i in range(n):
        transform[N, True](re1, im1, 1 << i)

#     for i in range(len(re)):
#         print(re1[i], "+ i *", im1[i], end=", ")
#     print("\n")

    var re2 = List[float_type](length=N, fill=0.0)
    re2[0] = 1.0
    var im2 = List[float_type](length=N, fill=0.0)


    for i in range(n):
        transform[N, False](re2, im2, 1 << i)

        #     for i in range(len(re)):
        #         print(re2[i], "+ i *", im2[i], end=", ")
        #     print("\n")

    for i in range(N):
        assert_almost_equal(re[i], re1[i])
        assert_almost_equal(im[i], im1[i])

        assert_almost_equal(re[i], re2[i])
        assert_almost_equal(im[i], im2[i])

    iters = 5

    print("n =", n, ", stride =", stride, ", iterations=", iters)
    t0 = benchmark.run[test_loop[N, stride]](iters).mean()
    print("loop", t0)
    t1 = benchmark.run[test_elementwise[N, stride]](iters).mean()
    print("elementwise", t1)
    print("speedup of elementwise over loop", t0/t1)
    t2 = benchmark.run[test_vectorize[N, stride]](iters).mean()
    print("vectorize", t2)
    print("speedup of vectorize over loop", t0/t2)