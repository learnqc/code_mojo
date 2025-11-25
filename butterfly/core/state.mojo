import random
import time
import math
from math import sqrt, cos, sin, log2, log10, atan2, floor
from testing import assert_true, assert_almost_equal

from sys.info import simd_width_of

from algorithm import vectorize, parallelize
from buffer import NDBuffer

from butterfly import *
from butterfly.core.gates import *

from algorithm import parallelize

alias dtype = Type
alias float_type = Scalar[dtype]

alias simd_width = simd_width_of[dtype]()

def init_state(n: Int) -> State:
    var state:State = [`1` if i == 0 else `0` for i in range(2 ** n)]
    return state^

def init_state_grid(row_bits: Int, col_bits: Int) -> GridState:
    R = 1 << row_bits
    C = 1 << col_bits
    grid_state = List[State](capacity=R)

    for _ in range(R):
        grid_state.append([`0` for _ in range(C)])
    grid_state[0][0] = `1`

    return grid_state^

def init_state_a[n: Int]() -> ArrayState[1<<n]:
    var state = ArrayState[1 << n](fill=`0`)
    state[0] = `1`
    return state^

def generate_state(n:Int, seed: Int = 555) -> State:
    # Choose a seed
    random.seed(seed)
    # Generate random probabilities that add up to 1
    var probs: List[FloatType] = [abs(random.random_float64(0, 1).cast[Type]()) for _ in range(2**n)]

    total: FloatType = 0.0
    for p in probs:
        total += p

    for i in range(len(probs)):
        probs[i] = probs[i]/total

    # Generate random angles in radians
    angles = [random.random_float64(0, 2 * math.pi).cast[Type]() for _ in range(2**n)]
    # Build the quantum state array
    state = [sqrt(p)*cis(theta) for (p, theta) in zip(probs, angles)]
    return state^

# @always_inline
fn process_pair(mut state: State, gate: Gate, k0: Int, k1: Int):
    x = state[k0]
    y = state[k1]
    # new amplitudes
    state[k0] = x * gate[0][0] + y * gate[0][1]
    state[k1] = x * gate[1][0] + y * gate[1][1]

fn transform[par: Int = 0](mut state: State, target: Int, gate: Gate):
    l = len(state)
    stride = 1 << target

    double_strides_per_work_item = l//(2*stride)//par
#     print(stride, double_strides_per_work_item)

    @parameter
    fn worker(j: Int):
        for k in range(double_strides_per_work_item):
            for idx in range(j*double_strides_per_work_item*2*stride + 2*k*stride, j*double_strides_per_work_item*2*stride + (2*k+1)*stride):
#                 print(j, idx, idx + stride)
                process_pair(state, gate, idx, idx + stride)

    @parameter
    fn worker1(j: Int):
        item_size = stride//par
#         print("work item", j, "of size", item_size)
        offsite = 2*stride*(j//par) + (j%par)*item_size
        for idx in range(offsite, offsite + item_size):
#             print(j, idx, idx + stride)
            process_pair(state, gate, idx, idx + stride)

    if par > 0:
        if double_strides_per_work_item > 0:
#             print("worker for target", target)
            parallelize[worker](par, par)
        elif stride >= par:
#             print("worker1 for target", target)
            parallelize[worker1](par*l//(2*stride), par)
        else:
            print("No parallelism for target", target)
#             r  = 0
#             for j in range(l//2):
#                 idx = 2*j - r     # r = j%stride
#                 process_pair(state, gate, idx, idx + stride)
#
#                 r += 1
#                 if r == stride:
#                     r = 0
            for k in range(l//(2*stride)):
                for idx in range(k*2*stride, k*2*stride + stride):
                    process_pair(state, gate, idx, idx + stride)

    else:
#         print("No parallelism for target", target)
        r  = 0
        for j in range(l//2):
            idx = 2*j - r     # r = j%stride
            process_pair(state, gate, idx, idx + stride)

            r += 1
            if r == stride:
                r = 0

#         for k in range(l//(2*stride)):
#             for idx in range(k*2*stride, k*2*stride + stride):
#                 process_pair(state, gate, idx, idx + stride)

fn transform_simd[N: Int](mut state: State, target: Int, gate: Gate):
    stride = 1 << target

    gate_re = [[gate[0][0].re, gate[0][1].re], [gate[1][0].re, gate[1][1].re]]
    gate_im = [[gate[0][0].im, gate[0][1].im], [gate[1][0].im, gate[1][1].im]]

    re = [a.re for a in state]
    im = [a.im for a in state]

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

    parallelize[worker_simd](num_work_items, num_threads)
    state = [Amplitude(re[i], im[i]) for i in range(N)]

fn transform_grid[par: Int = 0](mut state: GridState, target: Int, gate: Gate) raises:
    R = len(state)
    C = len(state[0])

    assert_true((1 << target) < R*C)

    col_bits = Int(log2(Float32(C)))

    @parameter
    fn row_worker(r: Int):
        transform(state[r], target, gate)

    @parameter
    fn column_worker(c: Int):
        t = target - col_bits
        stride = 1 << t

#         r  = 0
#         for j in range(R//2):
#             idx = 2*j - r     # r = j%stride
#             x = state[idx][c]
#             y = state[idx + stride][c]
#             # new amplitudes
#             state[idx][c] = x * gate[0][0] + y * gate[0][1]
#             state[idx + stride][c] = x * gate[1][0] + y * gate[1][1]
#
#             r += 1
#             if r == stride:
#                 r = 0

        for k in range(R//(2*stride)):
            for idx in range(k*2*stride, k*2*stride + stride):
                x = state[idx][c]
                y = state[idx + stride][c]
                # new amplitudes
                state[idx][c] = x * gate[0][0] + y * gate[0][1]
                state[idx + stride][c] = x * gate[1][0] + y * gate[1][1]

    if target < col_bits:
        if par > 0:
            parallelize[row_worker](R, par)
        else:
            for r in range(R):
#                 transform(state[r], target, gate)
                row_worker(r)
    else:
        if par > 0:
            parallelize[column_worker](C, par)
        else:
#             t = target - col_bits
#             stride = 1 << t
            for c in range(C):
                column_worker(c)
    #             r  = 0
    #             for j in range(R//2):
    #                 idx = 2*j - r     # r = j%stride
    #                 x = state[idx][c]
    #                 y = state[idx + stride][c]
    #                 # new amplitudes
    #                 state[idx][c] = x * gate[0][0] + y * gate[0][1]
    #                 state[idx + stride][c] = x * gate[1][0] + y * gate[1][1]
    #
    #                 r += 1
    #                 if r == stride:
    #                     r = 0

fn process_pair_a(mut state: ArrayState, gate: Gate, k0: Int, k1: Int):
    x = state[k0]
    y = state[k1]
    # new amplitudes
    state[k0] = x * gate[0][0] + y * gate[0][1]
    state[k1] = x * gate[1][0] + y * gate[1][1]

fn transform_a(mut state: ArrayState, target: Int, gate: Gate):
    l = len(state)
    stride = 1 << target
    r  = 0
    for j in range(l//2):
        idx = 2*j - r     # r = j%stride
        process_pair_a(state, gate, idx, idx + stride)

        r += 1
        if r == stride:
            r = 0

fn transform_swap(mut state: State, target: Int, gate: Gate):
    l = len(state)
    stride = 1 << target
    # swap
    r  = 0
    for j in range(0, l//4):
        idx = 4*j - r
        state.swap_elements(idx + 1, idx + stride)

        r += 2
        if r >= stride:
            r = 0

#    apply gate to consecutive entries
    for j in range(0, l//2):
        idx = 2*j
        process_pair(state, gate, idx, idx + 1)

#   swap back
    r  = 0
    for j in range(0, l//4):
        idx = 4*j - r
        state.swap_elements(idx + 1, idx + stride)

        r += 2
        if r >= stride:
            r = 0

def is_bit_set(m: Int, k: Int) -> Bool:
    return m & Int(1 << k) != 0

def c_transform(mut state: State, control: Int, target: Int, gate: Gate):
    stride = 1 << target
    r  = 0
    for j in range(len(state)//2):
        idx = 2*j - r     # r = j%stride
        if is_bit_set(Int(idx), control):
            process_pair(state, gate, Int(idx), Int(idx + stride))

        r += 1
        if r == stride:
            r = 0

def iqft(mut state: State, targets: List[Int]):
    for j in reversed(range(len(targets))):
        transform(state, targets[j], H)
        for k in reversed(range(j)):
            c_transform(state, targets[j], targets[k], P(-pi / 2 ** (j - k)))

def measure_qubit(mut state: State, t: Int, reset:Bool, v: Bool) -> Bool:
    # n = Int(log2(len(state)))

    stride = 1<<t

    prob0_sq: FloatType = 0
    prob1_sq: FloatType = 0

    r = 0
    for j in range(len(state)//2):
        idx = 2*j - r     # r = j%stride
        prob0_sq += state[idx].re*state[idx].re + state[idx].im*state[idx].im
        prob1_sq += state[idx + stride].re*state[idx + stride].re + state[idx + stride].im*state[idx + stride].im

        r += 1
        if r == stride:
            r = 0


    if v == False:
        r = 0
        for j in range(len(state)//2):
            idx = 2*j - r     # r = j%stride
            state[idx].re /= sqrt(prob0_sq)
            state[idx].im /= sqrt(prob0_sq)
            state[idx + stride] = `0`

            r += 1
            if r == stride:
                r = 0

    else:
        r = 0
        for j in range(len(state)//2):
            idx = 2*j - r     # r = j%stride
            state[idx] = `0`
            state[idx + stride].re /= sqrt(prob1_sq)
            state[idx + stride].im /= sqrt(prob1_sq)

            r += 1
            if r == stride:
                r = 0

        if reset:
            transform(state, t, X)

    return v




