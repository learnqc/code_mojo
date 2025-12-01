import random
import time
import math
from math import sqrt, cos, sin, log2, log10, atan2, floor
from testing import assert_true, assert_almost_equal
from sys.info import simd_width_of
from algorithm import vectorize, parallelize
from buffer import NDBuffer

from butterfly.core.types import *
from butterfly.core.gates import *

alias simd_width = simd_width_of[Type]()


struct QuantumState:
    var re: List[FloatType]
    var im: List[FloatType]

    fn __init__(out self, n: Int):
        self.re = List[FloatType](capacity=1 << n)
        self.im = List[FloatType](capacity=1 << n)
        self.re.append(1.0)
        self.im.append(0.0)
        for _ in range(1, 1 << n):
            self.re.append(0.0)
            self.im.append(0.0)

    fn __init__(out self, var re: List[FloatType], var im: List[FloatType]):
        self.re = re^
        self.im = im^

    fn __copyinit__(out self, existing: Self):
        self.re = List[FloatType](capacity=len(existing.re))
        self.im = List[FloatType](capacity=len(existing.im))
        for i in range(len(existing.re)):
            self.re.append(existing.re[i])
            self.im.append(existing.im[i])

    fn __moveinit__(out self, deinit existing: Self):
        self.re = existing.re^
        self.im = existing.im^

    fn __len__(self) -> Int:
        return len(self.re)

    fn size(self) -> Int:
        return len(self.re)

    fn __getitem__(self, idx: Int) -> Amplitude:
        return Amplitude(self.re[idx], self.im[idx])

    fn __setitem__(mut self, idx: Int, val: Amplitude):
        self.re[idx] = val.re
        self.im[idx] = val.im


alias State = QuantumState


def init_state(n: Int) -> QuantumState:
    var state = QuantumState(n)
    state[0] = `1`
    return state^


def init_state_grid(row_bits: Int, col_bits: Int) -> GridState:
    R = 1 << row_bits
    C = 1 << col_bits
    grid_state = List[List[Amplitude]](capacity=R)
    for _ in range(R):
        var row = List[Amplitude](capacity=C)
        for _ in range(C):
            row.append(`0`)
        grid_state.append(row^)
    grid_state[0][0] = `1`
    return grid_state^


def init_state_a[n: Int]() -> ArrayState[1 << n]:
    var state = ArrayState[1 << n](fill=`0`)
    state[0] = `1`
    return state^


def generate_state(n: Int, seed: Int = 555) -> QuantumState:
    random.seed(seed)
    var probs = List[FloatType](capacity=1 << n)
    for _ in range(1 << n):
        probs.append(abs(random.random_float64(0, 1).cast[Type]()))

    total: FloatType = 0.0
    for p in probs:
        total += p

    for i in range(len(probs)):
        probs[i] = probs[i] / total

    var state = QuantumState(n)
    for i in range(1 << n):
        theta = random.random_float64(0, 2 * math.pi).cast[Type]()
        state[i] = sqrt(probs[i]) * cis(theta)

    return state^


# @always_inline
fn process_pair(mut state: QuantumState, gate: Gate, k0: Int, k1: Int):
    x = state[k0]
    y = state[k1]
    # new amplitudes
    state[k0] = x * gate[0][0] + y * gate[0][1]
    state[k1] = x * gate[1][0] + y * gate[1][1]


fn transform[par: Int = 0](mut state: QuantumState, target: Int, gate: Gate):
    l = state.size()
    stride = 1 << target

    var double_strides_per_work_item = 0
    if par > 0:
        double_strides_per_work_item = l // (2 * stride) // par

    @parameter
    fn worker(j: Int):
        for k in range(double_strides_per_work_item):
            for idx in range(
                j * double_strides_per_work_item * 2 * stride + 2 * k * stride,
                j * double_strides_per_work_item * 2 * stride
                + (2 * k + 1) * stride,
            ):
                process_pair(state, gate, idx, idx + stride)

    @parameter
    fn worker1(j: Int):
        var item_size = 0
        if par > 0:
            item_size = stride // par
        offsite = 2 * stride * (j // par) + (j % par) * item_size
        for idx in range(offsite, offsite + item_size):
            process_pair(state, gate, idx, idx + stride)

    if par > 0:
        if double_strides_per_work_item > 0:
            parallelize[worker](par, par)
        elif stride >= par:
            parallelize[worker1](par * l // (2 * stride), par)
        else:
            print("No parallelism for target", target)
            for k in range(l // (2 * stride)):
                for idx in range(k * 2 * stride, k * 2 * stride + stride):
                    process_pair(state, gate, idx, idx + stride)

    else:
        r = 0
        for j in range(l // 2):
            idx = 2 * j - r  # r = j%stride
            process_pair(state, gate, idx, idx + stride)

            r += 1
            if r == stride:
                r = 0


fn transform_simd_base[
    N: Int
](mut state: QuantumState, stride: Int, gate: Gate):
    gate_re = [[gate[0][0].re, gate[0][1].re], [gate[1][0].re, gate[1][1].re]]
    gate_im = [[gate[0][0].im, gate[0][1].im], [gate[1][0].im, gate[1][1].im]]

    alias num_work_items = 8
    alias num_threads = num_work_items
    alias chunk_size = max(1, N // 2 // num_work_items)

    var vector_re = NDBuffer[Type, 1, _, N](state.re)
    var vector_im = NDBuffer[Type, 1, _, N](state.im)

    @always_inline
    @parameter
    fn butterfly_simd[simd_width: Int](idx: Int):
        zero_idx = 2 * idx - idx % stride
        one_idx = zero_idx + stride

        var elem0_re = vector_re.load[width=simd_width](zero_idx)
        var elem0_im = vector_im.load[width=simd_width](zero_idx)
        var elem1_re = vector_re.load[width=simd_width](one_idx)
        var elem1_im = vector_im.load[width=simd_width](one_idx)

        elem0_orig_re = elem0_re
        elem0_orig_im = elem0_im

        elem1_orig_re = elem1_re
        elem1_orig_im = elem1_im

        elem0_re = elem0_orig_re.fma(
            gate_re[0][0],
            -gate_im[0][0] * elem0_orig_im
            + elem1_orig_re.fma(gate_re[0][1], -gate_im[0][1] * elem1_orig_im),
        )
        elem0_im = elem0_orig_re.fma(
            gate_im[0][0],
            gate_re[0][0] * elem0_orig_im
            + elem1_orig_re.fma(gate_im[0][1], gate_re[0][1] * elem1_orig_im),
        )
        elem1_re = elem0_orig_re.fma(
            gate_re[1][0],
            -gate_im[1][0] * elem0_orig_im
            + elem1_orig_re.fma(gate_re[1][1], -gate_im[1][1] * elem1_orig_im),
        )
        elem1_im = elem0_orig_re.fma(
            gate_im[1][0],
            gate_re[1][0] * elem0_orig_im
            + elem1_orig_re.fma(gate_im[1][1], gate_re[1][1] * elem1_orig_im),
        )

        vector_re.store[width=simd_width](zero_idx, elem0_re)
        vector_im.store[width=simd_width](zero_idx, elem0_im)
        vector_re.store[width=simd_width](one_idx, elem1_re)
        vector_im.store[width=simd_width](one_idx, elem1_im)

    @parameter
    @always_inline
    fn worker_simd(item_id: Int):
        var start = item_id * chunk_size
        for idx in range(start, start + chunk_size, simd_width):
            butterfly_simd[simd_width](idx)

    parallelize[worker_simd](num_work_items, num_threads)


fn transform_simd[N: Int](mut state: QuantumState, target: Int, gate: Gate):
    stride = 1 << target

    if stride < simd_width:
        transform(state, target, gate)
    else:
        # Optimized: No copy needed!
        transform_simd_base[N](state, stride, gate)


fn transform_grid[
    par: Int = 0
](mut state: GridState, target: Int, gate: Gate) raises:
    R = len(state)
    C = len(state[0])

    assert_true((1 << target) < R * C)

    col_bits = Int(log2(Float32(C)))

    @parameter
    fn row_worker(r: Int):
        var re = [a.re for a in state[r]]
        var im = [a.im for a in state[r]]
        var qs = QuantumState(re.copy(), im.copy())
        transform(qs, target, gate)

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

        for k in range(R // (2 * stride)):
            for idx in range(k * 2 * stride, k * 2 * stride + stride):
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
    state[k0] = x * gate[0][0] + y * gate[0][1]
    state[k1] = x * gate[1][0] + y * gate[1][1]


fn transform_a(mut state: ArrayState, target: Int, gate: Gate):
    l = len(state)
    stride = 1 << target
    r = 0
    for j in range(l // 2):
        idx = 2 * j - r
        process_pair_a(state, gate, idx, idx + stride)
        r += 1
        if r == stride:
            r = 0


fn transform_swap(mut state: QuantumState, target: Int, gate: Gate):
    l = state.size()
    stride = 1 << target
    r = 0
    for j in range(0, l // 4):
        idx = 4 * j - r
        state.re.swap_elements(idx + 1, idx + stride)
        state.im.swap_elements(idx + 1, idx + stride)
        r += 2
        if r >= stride:
            r = 0

    for j in range(0, l // 2):
        idx = 2 * j
        process_pair(state, gate, idx, idx + 1)

    r = 0
    for j in range(0, l // 4):
        idx = 4 * j - r
        state.re.swap_elements(idx + 1, idx + stride)
        state.im.swap_elements(idx + 1, idx + stride)
        r += 2
        if r >= stride:
            r = 0


fn is_bit_set(m: Int, k: Int) -> Bool:
    return m & Int(1 << k) != 0


fn c_transform(mut state: QuantumState, control: Int, target: Int, gate: Gate):
    stride = 1 << target
    r = 0
    for j in range(state.size() // 2):
        idx = 2 * j - r
        if is_bit_set(Int(idx), control):
            process_pair(state, gate, Int(idx), Int(idx + stride))
        r += 1
        if r == stride:
            r = 0


fn c_transform_simd_base[
    N: Int
](mut state: QuantumState, control: Int, stride: Int, gate: Gate):
    gate_re = [[gate[0][0].re, gate[0][1].re], [gate[1][0].re, gate[1][1].re]]
    gate_im = [[gate[0][0].im, gate[0][1].im], [gate[1][0].im, gate[1][1].im]]

    alias num_work_items = 8
    alias num_threads = num_work_items
    alias chunk_size = max(1, N // 2 // num_work_items)

    var vector_re = NDBuffer[Type, 1, _, N](state.re)
    var vector_im = NDBuffer[Type, 1, _, N](state.im)

    @always_inline
    @parameter
    fn butterfly_simd[simd_width: Int](idx: Int):
        zero_idx = 2 * idx - idx % stride
        one_idx = zero_idx + stride

        var elem0_re = vector_re.load[width=simd_width](zero_idx)
        var elem0_im = vector_im.load[width=simd_width](zero_idx)
        var elem1_re = vector_re.load[width=simd_width](one_idx)
        var elem1_im = vector_im.load[width=simd_width](one_idx)

        # Construct indices vector: zero_idx, zero_idx+1, ...
        # We need to cast zero_idx to SIMD[DType.int64, simd_width] to add iota
        var offsets = SIMD[DType.int64, simd_width]()
        for i in range(simd_width):
            offsets[i] = i
        var indices = SIMD[DType.int64, simd_width](zero_idx) + offsets

        var mask_val = indices & (1 << control)
        var mask = mask_val.cast[DType.bool]()

        # Optimization: If no control bits set, skip
        # reduce_or seems to be missing or mask is inferred as Bool?
        # Commenting out optimization for now to ensure correctness of logic first.
        # if not mask.reduce_or():
        #    return

        elem0_orig_re = elem0_re
        elem0_orig_im = elem0_im

        elem1_orig_re = elem1_re
        elem1_orig_im = elem1_im

        new_elem0_re = elem0_orig_re.fma(
            gate_re[0][0],
            -gate_im[0][0] * elem0_orig_im
            + elem1_orig_re.fma(gate_re[0][1], -gate_im[0][1] * elem1_orig_im),
        )
        new_elem0_im = elem0_orig_re.fma(
            gate_im[0][0],
            gate_re[0][0] * elem0_orig_im
            + elem1_orig_re.fma(gate_im[0][1], gate_re[0][1] * elem1_orig_im),
        )
        new_elem1_re = elem0_orig_re.fma(
            gate_re[1][0],
            -gate_im[1][0] * elem0_orig_im
            + elem1_orig_re.fma(gate_re[1][1], -gate_im[1][1] * elem1_orig_im),
        )
        new_elem1_im = elem0_orig_re.fma(
            gate_im[1][0],
            gate_re[1][0] * elem0_orig_im
            + elem1_orig_re.fma(gate_im[1][1], gate_re[1][1] * elem1_orig_im),
        )

        # Blend results based on mask
        elem0_re = mask.select(new_elem0_re, elem0_re)
        elem0_im = mask.select(new_elem0_im, elem0_im)
        elem1_re = mask.select(new_elem1_re, elem1_re)
        elem1_im = mask.select(new_elem1_im, elem1_im)

        vector_re.store[width=simd_width](zero_idx, elem0_re)
        vector_im.store[width=simd_width](zero_idx, elem0_im)
        vector_re.store[width=simd_width](one_idx, elem1_re)
        vector_im.store[width=simd_width](one_idx, elem1_im)

    @parameter
    @always_inline
    fn worker_simd(item_id: Int):
        var start = item_id * chunk_size
        var end = min(start + chunk_size, N // 2)
        for idx in range(start, end, simd_width):
            butterfly_simd[simd_width](idx)

    parallelize[worker_simd](num_work_items, num_threads)


fn c_transform_simd[
    N: Int
](mut state: QuantumState, control: Int, target: Int, gate: Gate):
    stride = 1 << target
    if stride < simd_width:
        c_transform(state, control, target, gate)
    else:
        c_transform_simd_base[N](state, control, stride, gate)


def iqft(mut state: QuantumState, targets: List[Int]):
    for j in reversed(range(len(targets))):
        transform(state, targets[j], H)
        for k in reversed(range(j)):
            c_transform(state, targets[j], targets[k], P(-pi / 2 ** (j - k)))


def iqft_simd[N: Int](mut state: QuantumState, targets: List[Int]):
    for j in reversed(range(len(targets))):
        transform_simd[N](state, targets[j], H)
        for k in reversed(range(j)):
            c_transform_simd[N](
                state, targets[j], targets[k], P(-pi / 2 ** (j - k))
            )


def measure_qubit(
    mut state: QuantumState, t: Int, reset: Bool, v: Bool
) -> Bool:
    stride = 1 << t
    prob0_sq: FloatType = 0
    prob1_sq: FloatType = 0
    r = 0
    for j in range(state.size() // 2):
        idx = 2 * j - r
        # Access re/im directly for performance or use __getitem__
        # Using direct access for now as it's cleaner than constructing Amplitudes
        prob0_sq += (
            state.re[idx] * state.re[idx] + state.im[idx] * state.im[idx]
        )
        prob1_sq += (
            state.re[idx + stride] * state.re[idx + stride]
            + state.im[idx + stride] * state.im[idx + stride]
        )
        r += 1
        if r == stride:
            r = 0

    if v == False:
        r = 0
        for j in range(state.size() // 2):
            idx = 2 * j - r
            state.re[idx] /= sqrt(prob0_sq)
            state.im[idx] /= sqrt(prob0_sq)
            state[idx + stride] = `0`
            r += 1
            if r == stride:
                r = 0
    else:
        r = 0
        for j in range(state.size() // 2):
            idx = 2 * j - r
            state[idx] = `0`
            state.re[idx + stride] /= sqrt(prob1_sq)
            state.im[idx + stride] /= sqrt(prob1_sq)
            r += 1
            if r == stride:
                r = 0
        if reset:
            transform(state, t, X)
    return v
