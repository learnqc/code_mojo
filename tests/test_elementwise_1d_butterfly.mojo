from math import erf, exp, tanh
from sys.info import simd_width_of

from algorithm import elementwise, parallelize
from buffer import NDBuffer
from memory import UnsafePointer
from testing import assert_almost_equal
# from testing import TestSuite

from utils.index import IndexList

import benchmark

alias float_type = Float32
alias dtype = DType.float32

alias simd_width = simd_width_of[dtype]()

alias n = 25
alias target = n-1
alias stride = 1 << target

alias num_elements = 1 << n

alias num_work_items = 2
alias num_threads = num_work_items

fn transform_parallel[show: Bool=False]() :
#     var ptr = UnsafePointer[float_type].alloc(num_elements)
#     var ptr = InlineArray[float_type, num_elements](uninitialized=True)
    var ptr = List[float_type](capacity=num_elements)

    var vector = NDBuffer[dtype, 1, _, num_elements](ptr)

    for i in range(len(vector)):
        vector[i] = i

    var chunk_size = len(vector)//num_work_items

    @parameter
    @__copy_capture(vector, chunk_size)
    @always_inline
    fn worker(thread_id: Int):
        @always_inline
        @__copy_capture(vector)
        @parameter
        fn butterfly[
            simd_width: Int, rank: Int, alignment: Int = 1
        ](idx: IndexList[rank]):
            k = idx[0]*num_work_items + thread_id*simd_width
            zero_idx = 2*k - k%stride
            one_idx = zero_idx + stride
            if show:
                print("thread_id", thread_id, "k", k, "idx:", idx, ", pairs:", zero_idx, "--", zero_idx+simd_width, one_idx, "--", one_idx+simd_width)
            var elem0 = vector.load[width=simd_width](zero_idx)
            var elem1 = vector.load[width=simd_width](one_idx)
#             if show:
#                 print("loaded SIMDs:", elem0, elem1, "->", elem0 + elem1, elem0 - elem1)

            vector.store[width=simd_width](zero_idx, elem0 + elem1)
            vector.store[width=simd_width](one_idx, elem0 - elem1)

        try:
            elementwise[butterfly, simd_width](num_elements//2//num_work_items)
        except e:
            print("Caught an error:", e)

    parallelize[worker](num_work_items, num_threads)
#     for i in range(num_work_items):
#         worker(i)

#     print(vector)
#     if show:
#         print(vector)
#         for i in range(len(vector)):
#             print(vector[i], end=", ")
#         print("\n")
    #     ptr.free()

fn transform[show: Bool=False]() :
    #     var ptr = UnsafePointer[float_type].alloc(num_elements)
    #     var ptr = InlineArray[float_type, num_elements](uninitialized=True)
    var ptr = List[float_type](capacity=num_elements)

    var vector = NDBuffer[dtype, 1, _, num_elements](ptr)

    for i in range(len(vector)):
        vector[i] = i


    @always_inline
    @__copy_capture(vector)
    @parameter
    fn butterfly[
        simd_width: Int, rank: Int, alignment: Int = 1
    ](idx: IndexList[rank]):
        zero_idx = 2*idx[0] - idx[0]%stride
        one_idx = zero_idx + stride
        if show:
            print("idx:", idx, ", pairs:", zero_idx, "--", zero_idx+simd_width, one_idx, "--", one_idx+simd_width)
        var elem0 = vector.load[width=simd_width](zero_idx)
        var elem1 = vector.load[width=simd_width](one_idx)
        if show:
            print("loaded SIMDs:", elem0, elem1, "->", elem0 + elem1, elem0 - elem1)

        vector.store[width=simd_width](zero_idx, elem0 + elem1)
        vector.store[width=simd_width](one_idx, elem0 - elem1)

    try:
        elementwise[butterfly, simd_width](num_elements//2)
    except e:
        print("Caught an error:", e)

#     print(vector)

    if show:
        print(vector)
        for i in range(len(vector)):
            print(vector[i], end=", ")
        print("\n")

        #     ptr.free()

def main():
#     transform_parallel[False]()
#     transform[False]()

    iters = 10
    t0 = benchmark.run[transform[False]](iters).mean()
    print("elementwise", t0)
    t1 = benchmark.run[transform_parallel[False]](iters).mean()
    print("parallel elementwise", t1)
    print("speedup", t0/t1)
