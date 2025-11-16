
from math import erf, exp, tanh
from sys.info import simd_width_of

from algorithm import vectorize
from buffer import NDBuffer
from memory import UnsafePointer
from testing import assert_true

import benchmark


alias simd_width = simd_width_of[DType.float32]()

alias n = 30
alias target = n-1
alias stride = 1 << target

alias num_elements = 1 << n

def transform[show: Bool=False]():
    #     var ptr = UnsafePointer[Float32].alloc(num_elements)
#     var ptr = InlineArray[Float32, num_elements](uninitialized=True)
    var ptr = List[Float32](capacity=num_elements)

    var vector = NDBuffer[DType.float32, 1, _, num_elements](ptr)

    for i in range(len(vector)):
        vector[i] = i

    @parameter
    @__copy_capture(vector)
    @always_inline
    fn butterfly[simd_width: Int](idx: Int):
        zero_idx = 2*idx - idx%stride
        one_idx = zero_idx + stride
        if show:
            print("idx:", idx, ", pairs:", zero_idx, "--", zero_idx+simd_width, one_idx, "--", one_idx+simd_width)
        var elem0 = vector.load[width=simd_width](zero_idx)
        var elem1 = vector.load[width=simd_width](one_idx)
        if show:
            print("loaded SIMDs:", elem0, elem1, "->", elem0 + elem1, elem0 - elem1)

        vector.store[width=simd_width](zero_idx, elem0 + elem1)
        vector.store[width=simd_width](one_idx, elem0 - elem1)

    vectorize[butterfly, simd_width](len(vector)//2)

    if show:
        print(vector)
        for i in range(len(vector)):
            print(vector[i], end=", ")
        print("\n")

def main():
    assert_true(stride >= simd_width)
#     transform[True]()

    iters = 10
    t = benchmark.run[transform[False]](iters).mean()
    print(t)
