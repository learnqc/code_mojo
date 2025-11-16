import benchmark
from algorithm import parallelize
from buffer import NDBuffer

from butterfly.core.state import *

alias unit = benchmark.Unit.ms

alias type = DType.float32
alias scalar = Scalar[type]

fn test_traverse[n: Int, par: UInt = 0]():
    alias N = 1 << n

#     state = NDBuffer[type, 1, _, N](UnsafePointer[Scalar[type]].alloc(N))
#     for i in range(len(state)):
#                 state[i] = i

#     state = InlineArray[Int, N](fill=0)
#     for i in range(len(state)):
#         state[i] = i

#     state = [j for j in range(N)]
#     ret = [Int(0) for _ in range(N)]

#     var vector_stack = InlineArray[Scalar[DType.int], N](uninitialized=True)
    var vector_stack = List[Scalar[DType.int]](capacity=N)

    var state = NDBuffer[DType.int, 1, _, N](vector_stack)

    for i in range(len(state)):
        state[i] = i


    @parameter
    fn worker(j: Int):
        alias item_size = N//par
#         v = SIMD[DType.float32, item_size] ()
        # #     = state[j*item_size:(j+1)*item_size]
        for i in range(item_size):
            state[j*item_size + i] += state[N-1] # 1
#             v[i] = state[i + j*item_size]
#         v += 1
#         print(v)


#         for i in range(j*item_size, (j+1)*item_size):
#             state.swap_elements(i, i + 1)
#             ret[i] = state[i] + 1
#             state[i] += 1

    if par > 0:
        parallelize[worker](par, par)
    else:
        for i in range(len(state)):
#             state.swap_elements(i, i + 1)
#             ret[i] = state[i] + 1
            state[i] += state[N-1] # 1

#     print("par", par)
#     for i in range(len(state)):
#         print(state[i], end=",")
#     print("\n")

def main():
    alias n = 25
    alias iter = 5
    alias threads = 8
    traverse = benchmark.run[test_traverse[n]](iter).mean(unit)
    traverse_parallel = benchmark.run[test_traverse[n, threads]](iter).mean(unit)

    print("n =", n, ", threads =", threads)
    print("non-parallel", traverse)
    print("parallel", traverse_parallel)
    print("ratio", traverse/traverse_parallel)

# +=1
# n = 25 , threads = 8
# non-parallel 28.3967
# parallel 13.8606
# ratio 2.048735263985686

# += state[N-1]
# n = 25 , threads = 8
# non-parallel 26.8864
# parallel 17.773400000000002
# ratio 1.512732510380681