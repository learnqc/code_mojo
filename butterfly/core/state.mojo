import random
from bit.bit import bit_reverse
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


struct QuantumState(ImplicitlyCopyable, Sized):
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

    fn __iter__(self) -> _QuantumStateIterator:
        """Return an iterator over the amplitudes."""
        return _QuantumStateIterator(self)


struct _QuantumStateIterator:
    """Iterator for QuantumState that yields Amplitude values."""
    var state: QuantumState
    var index: Int

    fn __init__(out self, state: QuantumState):
        self.state = state
        self.index = 0

    fn __iter__(self) -> Self:
        return self

    fn __has_next__(self) -> Bool:
        return self.index < self.state.size()

    fn __next__(mut self) -> Amplitude:
        var result = self.state[self.index]
        self.index += 1
        return result

    fn __len__(self) -> Int:
        return self.state.size() - self.index


alias State = QuantumState


fn bit_reverse_state(mut state: QuantumState):
    n = Int(log2(Float64(state.size())))
    s_re = List[FloatType](length=1 << n, fill=0.0)
    s_im = List[FloatType](length=1 << n, fill=0.0)

    for i in range(1 << n):
        idx = Int(bit_reverse(SIMD[DType.uint64, 1](i))[0] >> (64 - n))
        s_re[i] = state.re[idx]
        s_im[i] = state.im[idx]

    state.re = s_re^
    state.im = s_im^


fn init_state(n: Int) -> QuantumState:
    var state = QuantumState(n)
    state[0] = `1`
    return state^


fn init_state_grid(row_bits: Int, col_bits: Int) -> GridState:
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


fn init_state_a[n: Int]() -> ArrayState[1 << n]:
    var state = ArrayState[1 << n](fill=`0`)
    state[0] = `1`
    return state^


fn generate_state(n: Int, seed: Int = 555) -> QuantumState:
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
            # no parallelism
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


fn transform_h(mut state: QuantumState, target: Int):
    l = state.size()
    stride = 1 << target
    r = 0
    for j in range(l // 2):
        idx = 2 * j - r  # r = j%stride
        state[idx] = (state[idx] + state[idx + stride]) * sq_half
        state[idx + stride] = state[idx] - state[idx + stride] * sq2

        r += 1
        if r == stride:
            r = 0


fn c_transform_interval_p(
    mut state: QuantumState, control: Int, target: Int, angle: FloatType
):
    var c_stride = 1 << control
    var t_stride = 1 << target
    var size = state.size()

    if target < control:
        # Iterate over control blocks
        # Each block is [k*2*c_stride + c_stride, k*2*c_stride + 2*c_stride)
        for k in range(size // (2 * c_stride)):
            var block_start = k * 2 * c_stride + c_stride
            # Inside the block, perform standard transform logic for target
            # Since c_stride > t_stride and block_start is a multiple of 2*t_stride,
            # we can just iterate over the sub-blocks of target.
            for j in range(c_stride // (2 * t_stride)):
                var sub_start = block_start + j * 2 * t_stride
                for idx in range(sub_start, sub_start + t_stride):
                    state[idx + t_stride] = state[idx + t_stride] * cis(angle)
    else:
        # target > control
        # Iterate over target blocks
        for k in range(size // (2 * t_stride)):
            var base = k * 2 * t_stride
            # Inside [base, base + t_stride), select indices with control bit 1
            # The control bit pattern repeats every 2*c_stride
            var num_periods = t_stride // (2 * c_stride)
            for p in range(num_periods):
                var p_start = base + p * 2 * c_stride + c_stride
                for idx in range(p_start, p_start + c_stride):
                    state[idx + t_stride] = state[idx + t_stride] * cis(angle)


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

    if stride < 4 * simd_width:  # TODO: check
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


fn mc_transform(
    mut state: QuantumState, controls: List[Int], target: Int, gate: Gate
):
    var mask = 0
    for i in range(len(controls)):
        mask |= 1 << controls[i]

    stride = 1 << target
    r = 0
    for j in range(state.size() // 2):
        idx = 2 * j - r
        # Check if all control bits are set (idx & mask == mask)
        if (Int(idx) & mask) == mask:
            process_pair(state, gate, Int(idx), Int(idx + stride))
        r += 1
        if r == stride:
            r = 0


fn mc_transform_interval(
    mut state: QuantumState, controls: List[Int], target: Int, gate: Gate
):
    if len(controls) == 0:
        transform(state, target, gate)
        return

    # Find the largest control bit to use for interval skipping
    var c_max = -1
    var mask = 0
    for i in range(len(controls)):
        var c = controls[i]
        mask |= 1 << c
        if c > c_max:
            c_max = c

    # We can reuse the logic from c_transform_interval for c_max,
    # but inside the inner loops, we must check the rest of the mask.
    # Actually, c_transform_interval logic is specific to ONE control.
    # We can adapt it: iterate blocks defined by c_max.

    var c_stride = 1 << c_max
    var t_stride = 1 << target
    var size = state.size()

    # Remaining mask excluding c_max (already implicitly checked by interval logic)
    var sub_mask = mask ^ (1 << c_max)

    if target < c_max:
        # Iterate over control blocks of c_max
        for k in range(size // (2 * c_stride)):
            var block_start = k * 2 * c_stride + c_stride
            for j in range(c_stride // (2 * t_stride)):
                var sub_start = block_start + j * 2 * t_stride
                for idx in range(sub_start, sub_start + t_stride):
                    # Check remaining controls
                    if (Int(idx) & sub_mask) == sub_mask:
                        process_pair(state, gate, idx, idx + t_stride)
    else:
        # target > c_max
        for k in range(size // (2 * t_stride)):
            var base = k * 2 * t_stride
            var num_periods = t_stride // (2 * c_stride)
            for p in range(num_periods):
                var p_start = base + p * 2 * c_stride + c_stride
                for idx in range(p_start, p_start + c_stride):
                    # Check remaining controls
                    if (Int(idx) & sub_mask) == sub_mask:
                        process_pair(state, gate, idx, idx + t_stride)


fn c_transform_interval(
    mut state: QuantumState, control: Int, target: Int, gate: Gate
):
    var c_stride = 1 << control
    var t_stride = 1 << target
    var size = state.size()

    if target < control:
        # Iterate over control blocks
        # Each block is [k*2*c_stride + c_stride, k*2*c_stride + 2*c_stride)
        for k in range(size // (2 * c_stride)):
            var block_start = k * 2 * c_stride + c_stride
            # Inside the block, perform standard transform logic for target
            # Since c_stride > t_stride and block_start is a multiple of 2*t_stride,
            # we can just iterate over the sub-blocks of target.
            for j in range(c_stride // (2 * t_stride)):
                var sub_start = block_start + j * 2 * t_stride
                for idx in range(sub_start, sub_start + t_stride):
                    process_pair(state, gate, idx, idx + t_stride)
    else:
        # target > control
        # Iterate over target blocks
        for k in range(size // (2 * t_stride)):
            var base = k * 2 * t_stride
            # Inside [base, base + t_stride), select indices with control bit 1
            # The control bit pattern repeats every 2*c_stride
            var num_periods = t_stride // (2 * c_stride)
            for p in range(num_periods):
                var p_start = base + p * 2 * c_stride + c_stride
                for idx in range(p_start, p_start + c_stride):
                    process_pair(state, gate, idx, idx + t_stride)


fn process_contiguous_simd[
    N: Int
](mut state: QuantumState, start: Int, count: Int, stride: Int, gate: Gate):
    var gate_re = [
        [gate[0][0].re, gate[0][1].re],
        [gate[1][0].re, gate[1][1].re],
    ]
    var gate_im = [
        [gate[0][0].im, gate[0][1].im],
        [gate[1][0].im, gate[1][1].im],
    ]

    var vector_re = NDBuffer[Type, 1, _, N](state.re)
    var vector_im = NDBuffer[Type, 1, _, N](state.im)

    @parameter
    fn butterfly_simd[simd_width: Int](i: Int):
        var zero_idx = start + i
        var one_idx = zero_idx + stride

        var elem0_re = vector_re.load[width=simd_width](zero_idx)
        var elem0_im = vector_im.load[width=simd_width](zero_idx)
        var elem1_re = vector_re.load[width=simd_width](one_idx)
        var elem1_im = vector_im.load[width=simd_width](one_idx)

        var elem0_orig_re = elem0_re
        var elem0_orig_im = elem0_im

        var elem1_orig_re = elem1_re
        var elem1_orig_im = elem1_im

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

    vectorize[butterfly_simd, simd_width](count)


fn _get_highest_bit(n: Int) -> Int:
    # Simple implementation for small n (qubit counts < 64)
    # Could utilize ctlz for optimization
    if n == 0:
        return -1
    var temp = n
    var pos = 0
    if temp >= 1 << 32:
        temp >>= 32
        pos += 32
    if temp >= 1 << 16:
        temp >>= 16
        pos += 16
    if temp >= 1 << 8:
        temp >>= 8
        pos += 8
    if temp >= 1 << 4:
        temp >>= 4
        pos += 4
    if temp >= 1 << 2:
        temp >>= 2
        pos += 2
    if temp >= 1 << 1:
        temp >>= 1
        pos += 1
    return pos


fn _mc_transform_simd_recursive[
    N: Int
](
    mut state: QuantumState,
    start: Int,
    count: Int,
    mask: Int,
    target: Int,
    gate: Gate,
):
    if mask == 0:
        # Base case: All controls satisfied, target bit resolved to 0 (implied)
        # Apply gate with SIMD. t_stride is 1 << target
        process_contiguous_simd[N](state, start, count, 1 << target, gate)
        return

    var c = _get_highest_bit(mask)
    var stride = 1 << c
    var remaining_mask = mask ^ stride

    # We iterate over blocks of size 2*stride
    # Because we are inside a recursion "block" of size 'count',
    # and 'count' MUST be a multiple of '2*stride' if we did this right?
    # Actually, start is aligned, count is power of 2.

    for block_start in range(start, start + count, 2 * stride):
        if c == target:
            # This is the target bit. We must process the '0' side.
            # Recursion enters the lower half: [block_start, block_start + stride)
            _mc_transform_simd_recursive[N](
                state, block_start, stride, remaining_mask, target, gate
            )
        else:
            # This is a control bit. We must process the '1' side.
            # Recursion enters the upper half: [block_start + stride, block_start + 2*stride)
            _mc_transform_simd_recursive[N](
                state,
                block_start + stride,
                stride,
                remaining_mask,
                target,
                gate,
            )


fn mc_transform_simd[
    N: Int
](mut state: QuantumState, controls: List[Int], target: Int, gate: Gate):
    var mask = 0
    for i in range(len(controls)):
        mask |= 1 << controls[i]

    # Add target to mask to handle it in the recursion hierarchy
    mask |= 1 << target

    # Start recursion covering the whole state
    _mc_transform_simd_recursive[N](state, 0, state.size(), mask, target, gate)


fn c_transform_interval_simd[
    N: Int
](mut state: QuantumState, control: Int, target: Int, gate: Gate):
    var c_stride = 1 << control
    var t_stride = 1 << target
    var size = state.size()

    # If the contiguous block size (t_stride) is too small for SIMD, fall back to scalar
    # Or if we just want to be safe. Let's say if t_stride < simd_width, we can't do the inner loop easily.
    if t_stride < simd_width:
        c_transform_interval(state, control, target, gate)
        return

    if target < control:
        # Iterate over control blocks
        for k in range(size // (2 * c_stride)):
            var block_start = k * 2 * c_stride + c_stride
            # Inside the block, iterate over sub-blocks
            for j in range(c_stride // (2 * t_stride)):
                var sub_start = block_start + j * 2 * t_stride
                # Here, [sub_start, sub_start + t_stride) is a contiguous block of pairs
                # with indices (idx, idx + t_stride).
                # We can vectorize this loop:
                # for idx in range(sub_start, sub_start + t_stride): ...
                process_contiguous_simd[N](
                    state, sub_start, t_stride, t_stride, gate
                )
    else:
        # target > control
        for k in range(size // (2 * t_stride)):
            var base = k * 2 * t_stride
            var num_periods = t_stride // (2 * c_stride)
            for p in range(num_periods):
                var p_start = base + p * 2 * c_stride + c_stride
                # Here [p_start, p_start + c_stride) is a contiguous block
                # where control bit is 1.
                # Pairs are (idx, idx + t_stride).
                process_contiguous_simd[N](
                    state, p_start, c_stride, t_stride, gate
                )


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

        #         new_elem0_re = elem0_orig_re.fma(
        #             gate_re[0][0],
        #             -gate_im[0][0] * elem0_orig_im
        #             + elem1_orig_re.fma(gate_re[0][1], -gate_im[0][1] * elem1_orig_im),
        #         )
        #         new_elem0_im = elem0_orig_re.fma(
        #             gate_im[0][0],
        #             gate_re[0][0] * elem0_orig_im
        #             + elem1_orig_re.fma(gate_im[0][1], gate_re[0][1] * elem1_orig_im),
        #         )
        #         new_elem1_re = elem0_orig_re.fma(
        #             gate_re[1][0],
        #             -gate_im[1][0] * elem0_orig_im
        #             + elem1_orig_re.fma(gate_re[1][1], -gate_im[1][1] * elem1_orig_im),
        #         )
        #         new_elem1_im = elem0_orig_re.fma(
        #             gate_im[1][0],
        #             gate_re[1][0] * elem0_orig_im
        #             + elem1_orig_re.fma(gate_im[1][1], gate_re[1][1] * elem1_orig_im),
        #         )

        # Blend results based on mask
        elem0_re = mask.select(
            elem0_orig_re.fma(
                gate_re[0][0],
                -gate_im[0][0] * elem0_orig_im
                + elem1_orig_re.fma(
                    gate_re[0][1], -gate_im[0][1] * elem1_orig_im
                ),
            ),
            elem0_re,
        )
        elem0_im = mask.select(
            elem0_orig_re.fma(
                gate_im[0][0],
                gate_re[0][0] * elem0_orig_im
                + elem1_orig_re.fma(
                    gate_im[0][1], gate_re[0][1] * elem1_orig_im
                ),
            ),
            elem0_im,
        )
        elem1_re = mask.select(
            elem0_orig_re.fma(
                gate_re[1][0],
                -gate_im[1][0] * elem0_orig_im
                + elem1_orig_re.fma(
                    gate_re[1][1], -gate_im[1][1] * elem1_orig_im
                ),
            ),
            elem1_re,
        )
        elem1_im = mask.select(
            elem0_orig_re.fma(
                gate_im[1][0],
                gate_re[1][0] * elem0_orig_im
                + elem1_orig_re.fma(
                    gate_im[1][1], gate_re[1][1] * elem1_orig_im
                ),
            ),
            elem1_im,
        )

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
    if stride < 4 * simd_width:  # TODO: check
        c_transform(state, control, target, gate)
    else:
        c_transform_simd_base[N](state, control, stride, gate)


fn iqft(mut state: QuantumState, targets: List[Int], swap: Bool = False):
    for j in reversed(range(len(targets))):
        transform(state, targets[j], H)
        for k in reversed(range(j)):
            c_transform(state, targets[j], targets[k], P(-pi / 2 ** (j - k)))

    if swap:
        bit_reverse_state(state)


fn iqft_interval(
    mut state: QuantumState, targets: List[Int], swap: Bool = False
):
    for j in reversed(range(len(targets))):
        # transform(state, targets[j], H)
        transform_h(state, targets[j])
        for k in reversed(range(j)):
            c_transform_interval_p(
                state, targets[j], targets[k], -pi / 2 ** (j - k)
            )

    if swap:
        bit_reverse_state(state)


fn iqft_simd[
    N: Int
](mut state: QuantumState, targets: List[Int], swap: Bool = False):
    for j in reversed(range(len(targets))):
        transform_simd[N](state, targets[j], H)
        for k in reversed(range(j)):
            c_transform_simd[N](
                state, targets[j], targets[k], P(-pi / 2 ** (j - k))
            )

    if swap:
        bit_reverse_state(state)


fn iqft_simd_interval[
    N: Int
](mut state: QuantumState, targets: List[Int], swap: Bool = False):
    for j in reversed(range(len(targets))):
        transform_simd[N](state, targets[j], H)
        for k in reversed(range(j)):
            c_transform_interval_simd[N](
                state, targets[j], targets[k], P(-pi / 2 ** (j - k))
            )

    if swap:
        bit_reverse_state(state)


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
