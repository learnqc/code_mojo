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
from butterfly.utils.config import get_workers

alias simd_width = simd_width_of[Type]()


struct QuantumState(Copyable, ImplicitlyCopyable, Movable, Sized):
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
        self.state = state.copy()
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


fn bit_reverse_state(mut state: QuantumState, parallel: Bool = True):
    """
    Bit Reversal of the quantum state.

    Args:
        state: The QuantumState to permute.
        parallel: If True (default), uses optimized SIMD/Parallel implementation.
                  If False, uses sequential scalar implementation (low memory overhead).
    """
    var n = state.size()
    var log_n = Int(log2(Float64(n)))

    var s_re = List[FloatType](length=n, fill=0.0)
    var s_im = List[FloatType](length=n, fill=0.0)

    if parallel:
        # Optimized Parallel SIMD
        var ptr_in_re = state.re.unsafe_ptr()
        var ptr_in_im = state.im.unsafe_ptr()
        var ptr_out_re = s_re.unsafe_ptr()
        var ptr_out_im = s_im.unsafe_ptr()

        @parameter
        fn worker(idx: Int):
            alias width = simd_width
            var base = idx * width

            # Vectorized Index Generation
            var offsets = SIMD[DType.uint64, width]()
            for i in range(width):
                offsets[i] = i
            var vec_idx = SIMD[DType.uint64, width](base) + offsets

            # Bit Reverse
            var r_idx_u64 = bit_reverse(vec_idx) >> (64 - log_n)
            var r_idx = r_idx_u64.cast[DType.int64]()

            var val_re = ptr_in_re.gather(r_idx)
            var val_im = ptr_in_im.gather(r_idx)

            ptr_out_re.store(base, val_re)
            ptr_out_im.store(base, val_im)

        parallelize[worker](n // simd_width)
    else:
        # Sequential Scalar (Fallback)
        # Using unsafe pointers for performance even in sequential mode
        var ptr_in_re = state.re.unsafe_ptr()
        var ptr_in_im = state.im.unsafe_ptr()
        var ptr_out_re = s_re.unsafe_ptr()
        var ptr_out_im = s_im.unsafe_ptr()

        for i in range(n):
            var r_idx = Int(
                bit_reverse(SIMD[DType.uint64, 1](i))[0] >> (64 - log_n)
            )
            ptr_out_re[i] = ptr_in_re[r_idx]
            ptr_out_im[i] = ptr_in_im[r_idx]

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
    var u_re = state.re[k0]
    var u_im = state.im[k0]
    var v_re = state.re[k1]
    var v_im = state.im[k1]

    # Matrix multiplication
    state.re[k0] = (
        u_re * gate[0][0].re
        - u_im * gate[0][0].im
        + v_re * gate[0][1].re
        - v_im * gate[0][1].im
    )
    state.im[k0] = (
        u_re * gate[0][0].im
        + u_im * gate[0][0].re
        + v_re * gate[0][1].im
        + v_im * gate[0][1].re
    )
    state.re[k1] = (
        u_re * gate[1][0].re
        - u_im * gate[1][0].im
        + v_re * gate[1][1].re
        - v_im * gate[1][1].im
    )
    state.im[k1] = (
        u_re * gate[1][0].im
        + u_im * gate[1][0].re
        + v_re * gate[1][1].im
        + v_im * gate[1][1].re
    )


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
    var l = state.size()
    var stride = 1 << target
    var r = 0
    for j in range(l // 2):
        var idx = 2 * j - r  # r = j%stride
        var u = state[idx]
        var v = state[idx + stride]
        state[idx] = (u + v) * sq_half
        state[idx + stride] = (u - v) * sq_half

        r += 1
        if r == stride:
            r = 0


fn transform_h_block_style(mut state: QuantumState, target: Int):
    """Block-based implementation avoiding modulo.

    Processes blocks of 2*stride elements, computing butterflies
    for the first stride elements in each block.
    """
    var l = state.size()
    var stride = 1 << target
    var scale = FloatType(1.0 / sqrt(2.0))

    for k in range(l // (2 * stride)):
        for idx in range(k * 2 * stride, k * 2 * stride + stride):
            # Load u and v
            var u_re = state.re[idx]
            var u_im = state.im[idx]
            var v_re = state.re[idx + stride]
            var v_im = state.im[idx + stride]

            # Butterfly: (u+v)/√2 and (u-v)/√2
            state.re[idx] = (u_re + v_re) * scale
            state.im[idx] = (u_im + v_im) * scale
            state.re[idx + stride] = (u_re - v_re) * scale
            state.im[idx + stride] = (u_im - v_im) * scale


fn transform_h_simd(mut state: QuantumState, target: Int):
    """Vectorized Hadamard gate."""
    var l = state.size()
    var stride = 1 << target
    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()
    alias sq_half = 0.7071067811865476

    for k in range(0, l, 2 * stride):

        @parameter
        fn vectorize_h[width: Int](m: Int):
            var idx = k + m
            var u_re = ptr_re.load[width=width](idx)
            var u_im = ptr_im.load[width=width](idx)
            var v_re = ptr_re.load[width=width](idx + stride)
            var v_im = ptr_im.load[width=width](idx + stride)

            var sum_re = (u_re + v_re) * sq_half
            var sum_im = (u_im + v_im) * sq_half
            var diff_re = (u_re - v_re) * sq_half
            var diff_im = (u_im - v_im) * sq_half

            ptr_re.store[width=width](idx, sum_re)
            ptr_im.store[width=width](idx, sum_im)
            ptr_re.store[width=width](idx + stride, diff_re)
            ptr_im.store[width=width](idx + stride, diff_im)

        if stride >= simd_width:
            vectorize[vectorize_h, simd_width](stride)
        else:
            for m in range(stride):
                var idx = k + m
                var u_re = ptr_re[idx]
                var u_im = ptr_im[idx]
                var v_re = ptr_re[idx + stride]
                var v_im = ptr_im[idx + stride]
                ptr_re[idx] = (u_re + v_re) * sq_half
                ptr_im[idx] = (u_im + v_im) * sq_half
                ptr_re[idx + stride] = (u_re - v_re) * sq_half
                ptr_im[idx + stride] = (u_im - v_im) * sq_half


fn transform_h_simd_v2(mut state: QuantumState, target: Int):
    """Parallelized Vectorized Hadamard gate."""
    var l = state.size()
    var stride = 1 << target
    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()
    alias sq_half = 0.7071067811865476

    var num_work_items = get_workers("quantum_simd_v2_chunks")
    if num_work_items == 0:
        num_work_items = 16

    var total_blocks = l // (2 * stride)
    var actual_workers = min(num_work_items, total_blocks)
    var blocks_per_worker = max(1, total_blocks // num_work_items)

    @parameter
    fn worker(item_id: Int):
        var start_block = item_id * blocks_per_worker
        var end_block = (
            total_blocks if item_id
            == actual_workers - 1 else (item_id + 1) * blocks_per_worker
        )

        for k_idx in range(start_block, end_block):
            var k = k_idx * 2 * stride

            @parameter
            fn vectorize_h[width: Int](m: Int):
                var idx = k + m
                var u_re = ptr_re.load[width=width](idx)
                var u_im = ptr_im.load[width=width](idx)
                var v_re = ptr_re.load[width=width](idx + stride)
                var v_im = ptr_im.load[width=width](idx + stride)

                var sum_re = (u_re + v_re) * sq_half
                var sum_im = (u_im + v_im) * sq_half
                var diff_re = (u_re - v_re) * sq_half
                var diff_im = (u_im - v_im) * sq_half

                ptr_re.store[width=width](idx, sum_re)
                ptr_im.store[width=width](idx, sum_im)
                ptr_re.store[width=width](idx + stride, diff_re)
                ptr_im.store[width=width](idx + stride, diff_im)

            if stride >= simd_width:
                vectorize[vectorize_h, simd_width](stride)
            else:
                for m in range(stride):
                    var idx = k + m
                    var u_re = ptr_re[idx]
                    var u_im = ptr_im[idx]
                    var v_re = ptr_re[idx + stride]
                    var v_im = ptr_im[idx + stride]
                    ptr_re[idx] = (u_re + v_re) * sq_half
                    ptr_im[idx] = (u_im + v_im) * sq_half
                    ptr_re[idx + stride] = (u_re - v_re) * sq_half
                    ptr_im[idx + stride] = (u_im - v_im) * sq_half

    if actual_workers > 1:
        parallelize[worker](actual_workers)
    else:
        worker(0)


fn transform_z_simd(mut state: QuantumState, target: Int):
    """Vectorized Z gate."""
    var l = state.size()
    var stride = 1 << target
    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()

    for k in range(0, l, 2 * stride):

        @parameter
        fn vectorize_z[width: Int](m: Int):
            var idx = k + m + stride
            ptr_re.store[width=width](idx, -ptr_re.load[width=width](idx))
            ptr_im.store[width=width](idx, -ptr_im.load[width=width](idx))

        if stride >= simd_width:
            vectorize[vectorize_z, simd_width](stride)
        else:
            for m in range(stride):
                var idx = k + m + stride
                ptr_re[idx] = -ptr_re[idx]
                ptr_im[idx] = -ptr_im[idx]


fn transform_p_simd(mut state: QuantumState, target: Int, theta: Float64):
    """Vectorized Phase gate."""
    var l = state.size()
    var stride = 1 << target
    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()
    var cos_t = cos(theta)
    var sin_t = sin(theta)

    for k in range(0, l, 2 * stride):

        @parameter
        fn vectorize_p[width: Int](m: Int):
            var idx = k + m + stride
            var v_re = ptr_re.load[width=width](idx)
            var v_im = ptr_im.load[width=width](idx)
            ptr_re.store[width=width](idx, v_re * cos_t - v_im * sin_t)
            ptr_im.store[width=width](idx, v_re * sin_t + v_im * cos_t)

        if stride >= simd_width:
            vectorize[vectorize_p, simd_width](stride)
        else:
            for m in range(stride):
                var idx = k + m + stride
                var v_re = ptr_re[idx]
                var v_im = ptr_im[idx]
                ptr_re[idx] = v_re * cos_t - v_im * sin_t
                ptr_im[idx] = v_im * cos_t + v_re * sin_t


fn transform_x_simd(mut state: QuantumState, target: Int):
    """Vectorized X gate."""
    var l = state.size()
    var stride = 1 << target
    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()

    for k in range(0, l, 2 * stride):

        @parameter
        fn vectorize_x[width: Int](m: Int):
            var idx0 = k + m
            var idx1 = idx0 + stride
            var u_re = ptr_re.load[width=width](idx0)
            var u_im = ptr_im.load[width=width](idx0)
            var v_re = ptr_re.load[width=width](idx1)
            var v_im = ptr_im.load[width=width](idx1)
            ptr_re.store[width=width](idx0, v_re)
            ptr_im.store[width=width](idx0, v_im)
            ptr_re.store[width=width](idx1, u_re)
            ptr_im.store[width=width](idx1, u_im)

        if stride >= simd_width:
            vectorize[vectorize_x, simd_width](stride)
        else:
            for m in range(stride):
                var idx0 = k + m
                var idx1 = idx0 + stride
                var u_re = ptr_re[idx0]
                var u_im = ptr_im[idx0]
                var v_re = ptr_re[idx1]
                var v_im = ptr_im[idx1]
                ptr_re[idx0] = v_re
                ptr_im[idx0] = v_im
                ptr_re[idx1] = u_re
                ptr_im[idx1] = u_im


fn transform_x_simd_v2(mut state: QuantumState, target: Int):
    """Parallelized Vectorized X gate."""
    var l = state.size()
    var stride = 1 << target
    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()

    var num_work_items = get_workers("quantum_simd_v2_chunks")
    if num_work_items == 0:
        num_work_items = 16

    var total_blocks = l // (2 * stride)
    var actual_workers = min(num_work_items, total_blocks)
    var blocks_per_worker = max(1, total_blocks // num_work_items)

    @parameter
    fn worker(item_id: Int):
        var start_block = item_id * blocks_per_worker
        var end_block = (
            total_blocks if item_id
            == actual_workers - 1 else (item_id + 1) * blocks_per_worker
        )

        for k_idx in range(start_block, end_block):
            var k = k_idx * 2 * stride

            @parameter
            fn vectorize_x[width: Int](m: Int):
                var idx0 = k + m
                var idx1 = idx0 + stride
                var u_re = ptr_re.load[width=width](idx0)
                var u_im = ptr_im.load[width=width](idx0)
                var v_re = ptr_re.load[width=width](idx1)
                var v_im = ptr_im.load[width=width](idx1)
                ptr_re.store[width=width](idx0, v_re)
                ptr_im.store[width=width](idx0, v_im)
                ptr_re.store[width=width](idx1, u_re)
                ptr_im.store[width=width](idx1, u_im)

            if stride >= simd_width:
                vectorize[vectorize_x, simd_width](stride)
            else:
                for m in range(stride):
                    var idx0 = k + m
                    var idx1 = idx0 + stride
                    var u_re = ptr_re[idx0]
                    var u_im = ptr_im[idx0]
                    var v_re = ptr_re[idx1]
                    var v_im = ptr_im[idx1]
                    ptr_re[idx0] = v_re
                    ptr_im[idx0] = v_im
                    ptr_re[idx1] = u_re
                    ptr_im[idx1] = u_im

    if actual_workers > 1:
        parallelize[worker](actual_workers)
    else:
        worker(0)


fn transform_z_simd_v2(mut state: QuantumState, target: Int):
    """Parallelized Vectorized Z gate."""
    var l = state.size()
    var stride = 1 << target
    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()

    var num_work_items = get_workers("quantum_simd_v2_chunks")
    if num_work_items == 0:
        num_work_items = 16

    var total_blocks = l // (2 * stride)
    var actual_workers = min(num_work_items, total_blocks)
    var blocks_per_worker = max(1, total_blocks // num_work_items)

    @parameter
    fn worker(item_id: Int):
        var start_block = item_id * blocks_per_worker
        var end_block = (
            total_blocks if item_id
            == actual_workers - 1 else (item_id + 1) * blocks_per_worker
        )

        for k_idx in range(start_block, end_block):
            var k = k_idx * 2 * stride

            @parameter
            fn vectorize_z[width: Int](m: Int):
                var idx = k + m + stride
                ptr_re.store[width=width](idx, -ptr_re.load[width=width](idx))
                ptr_im.store[width=width](idx, -ptr_im.load[width=width](idx))

            if stride >= simd_width:
                vectorize[vectorize_z, simd_width](stride)
            else:
                for m in range(stride):
                    var idx = k + m + stride
                    ptr_re[idx] = -ptr_re[idx]
                    ptr_im[idx] = -ptr_im[idx]

    if actual_workers > 1:
        parallelize[worker](actual_workers)
    else:
        worker(0)


fn transform_p_simd_v2(mut state: QuantumState, target: Int, theta: Float64):
    """Parallelized Vectorized Phase gate."""
    var l = state.size()
    var stride = 1 << target
    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()
    var cos_t = cos(theta)
    var sin_t = sin(theta)

    var num_work_items = get_workers("quantum_simd_v2_chunks")
    if num_work_items == 0:
        num_work_items = 16

    var total_blocks = l // (2 * stride)
    var actual_workers = min(num_work_items, total_blocks)
    var blocks_per_worker = max(1, total_blocks // num_work_items)

    @parameter
    fn worker(item_id: Int):
        var start_block = item_id * blocks_per_worker
        var end_block = (
            total_blocks if item_id
            == actual_workers - 1 else (item_id + 1) * blocks_per_worker
        )

        for k_idx in range(start_block, end_block):
            var k = k_idx * 2 * stride

            @parameter
            fn vectorize_p[width: Int](m: Int):
                var idx = k + m + stride
                var v_re = ptr_re.load[width=width](idx)
                var v_im = ptr_im.load[width=width](idx)
                ptr_re.store[width=width](idx, v_re * cos_t - v_im * sin_t)
                ptr_im.store[width=width](idx, v_re * sin_t + v_im * cos_t)

            if stride >= simd_width:
                vectorize[vectorize_p, simd_width](stride)
            else:
                for m in range(stride):
                    var idx = k + m + stride
                    var v_re = ptr_re[idx]
                    var v_im = ptr_im[idx]
                    ptr_re[idx] = v_re * cos_t - v_im * sin_t
                    ptr_im[idx] = v_re * sin_t + v_im * cos_t

    if actual_workers > 1:
        parallelize[worker](actual_workers)
    else:
        worker(0)


fn c_transform_interval_p(
    mut state: QuantumState, control: Int, target: Int, angle: FloatType
):
    var c_stride = 1 << control
    var t_stride = 1 << target
    var size = state.size()

    z = cis(angle)
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
                    state[idx + t_stride] = state[idx + t_stride] * z
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
                    state[idx + t_stride] = state[idx + t_stride] * z


fn c_transform_interval_p_precomputed(
    mut state: QuantumState, control: Int, target: Int, angle: FloatType
):
    """FFT-style implementation with precomputed twiddle factor."""
    var c_stride = 1 << control
    var t_stride = 1 << target
    var size = state.size()

    # Precompute twiddle factor ONCE
    var w_re = cos(angle)
    var w_im = sin(angle)

    if target < control:
        for k in range(size // (2 * c_stride)):
            var block_start = k * 2 * c_stride + c_stride
            for j in range(c_stride // (2 * t_stride)):
                var sub_start = block_start + j * 2 * t_stride
                for idx in range(sub_start, sub_start + t_stride):
                    var re = state.re[idx + t_stride]
                    var im = state.im[idx + t_stride]
                    state.re[idx + t_stride] = re * w_re - im * w_im
                    state.im[idx + t_stride] = re * w_im + im * w_re
    else:
        for k in range(size // (2 * t_stride)):
            var base = k * 2 * t_stride
            var num_periods = t_stride // (2 * c_stride)
            for p in range(num_periods):
                var p_start = base + p * 2 * c_stride + c_stride
                for idx in range(p_start, p_start + c_stride):
                    var re = state.re[idx + t_stride]
                    var im = state.im[idx + t_stride]
                    state.re[idx + t_stride] = re * w_re - im * w_im
                    state.im[idx + t_stride] = re * w_im + im * w_re


fn transform_simd_base[
    N: Int
](mut state: QuantumState, stride: Int, gate: Gate):
    gate_re = [[gate[0][0].re, gate[0][1].re], [gate[1][0].re, gate[1][1].re]]
    gate_im = [[gate[0][0].im, gate[0][1].im], [gate[1][0].im, gate[1][1].im]]

    alias num_work_items = 8
    alias num_threads = num_work_items
    alias chunk_size = max(1, N // 2 // num_work_items)

    var vector_re = NDBuffer[Type, 1, _, N](state.re.unsafe_ptr())
    var vector_im = NDBuffer[Type, 1, _, N](state.im.unsafe_ptr())

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
    var stride = 1 << target

    if stride < 64:  # Force scalar for small strides/states
        transform(state, target, gate)
    elif is_h(gate):
        transform_h_simd(state, target)
    elif is_x(gate):
        transform_x_simd(state, target)
    elif is_z(gate):
        transform_z_simd(state, target)
    elif is_p(gate):
        transform_p_simd(state, target, get_phase_angle(gate))
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

    var vector_re = NDBuffer[Type, 1, _, N](state.re.unsafe_ptr())
    var vector_im = NDBuffer[Type, 1, _, N](state.im.unsafe_ptr())

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
                process_contiguous_simd[N](
                    state, p_start, c_stride, t_stride, gate
                )


fn c_transform_simd_base[
    N: Int
](mut state: QuantumState, control: Int, stride: Int, gate: Gate):
    var gate_re = [
        [gate[0][0].re, gate[0][1].re],
        [gate[1][0].re, gate[1][1].re],
    ]
    var gate_im = [
        [gate[0][0].im, gate[0][1].im],
        [gate[1][0].im, gate[1][1].im],
    ]

    alias num_work_items = 8
    alias num_threads = num_work_items
    alias chunk_size = max(1, N // 2 // num_work_items)

    var vector_re = NDBuffer[Type, 1, _, N](state.re.unsafe_ptr())
    var vector_im = NDBuffer[Type, 1, _, N](state.im.unsafe_ptr())

    @always_inline
    @parameter
    fn butterfly_simd[simd_width: Int](idx: Int):
        var zero_idx = 2 * idx - idx % stride
        var one_idx = zero_idx + stride

        var elem0_re = vector_re.load[width=simd_width](zero_idx)
        var elem0_im = vector_im.load[width=simd_width](zero_idx)
        var elem1_re = vector_re.load[width=simd_width](one_idx)
        var elem1_im = vector_im.load[width=simd_width](one_idx)

        # Precompute offsets for mask indexing? No, just keep simple for now.
        var offsets = SIMD[DType.int64, simd_width]()
        for i in range(simd_width):
            offsets[i] = i
        var indices = SIMD[DType.int64, simd_width](zero_idx) + offsets

        var mask_val = indices & (1 << control)
        var mask = mask_val.cast[DType.bool]()

        var elem0_orig_re = elem0_re
        var elem0_orig_im = elem0_im
        var elem1_orig_re = elem1_re
        var elem1_orig_im = elem1_im

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
    fn worker_simd_gold(item_id: Int):
        var start = item_id * chunk_size
        var end = min(start + chunk_size, N // 2)
        for idx in range(start, end, simd_width):
            butterfly_simd[simd_width](idx)

    parallelize[worker_simd_gold](num_work_items, num_threads)


fn c_transform_simd_base_v2[
    N: Int
](mut state: QuantumState, control: Int, stride: Int, gate: Gate):
    var gate_re = [
        [gate[0][0].re, gate[0][1].re],
        [gate[1][0].re, gate[1][1].re],
    ]
    var gate_im = [
        [gate[0][0].im, gate[0][1].im],
        [gate[1][0].im, gate[1][1].im],
    ]

    alias num_work_items = 8
    alias num_threads = num_work_items
    alias chunk_size = max(1, N // 2 // num_work_items)

    var vector_re = NDBuffer[Type, 1, _, N](state.re.unsafe_ptr())
    var vector_im = NDBuffer[Type, 1, _, N](state.im.unsafe_ptr())

    @always_inline
    @parameter
    fn butterfly_simd[simd_width: Int](idx: Int):
        var lower_mask = stride - 1
        var zero_idx = ((idx & ~lower_mask) << 1) | (idx & lower_mask)
        var one_idx = zero_idx + stride

        var elem0_re = vector_re.load[width=simd_width](zero_idx)
        var elem0_im = vector_im.load[width=simd_width](zero_idx)
        var elem1_re = vector_re.load[width=simd_width](one_idx)
        var elem1_im = vector_im.load[width=simd_width](one_idx)

        var offsets = SIMD[DType.int64, simd_width]()
        for i in range(simd_width):
            offsets[i] = i
        var indices = SIMD[DType.int64, simd_width](zero_idx) + offsets

        var control_mask = 1 << control
        var mask_val = indices & control_mask
        var mask = mask_val.cast[DType.bool]()

        var elem0_orig_re = elem0_re
        var elem0_orig_im = elem0_im
        var elem1_orig_re = elem1_re
        var elem1_orig_im = elem1_im

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
    fn worker_simd_v2(item_id: Int):
        var start = item_id * chunk_size
        var end = min(start + chunk_size, N // 2)
        for idx in range(start, end, simd_width):
            butterfly_simd[simd_width](idx)

    parallelize[worker_simd_v2](num_work_items, num_threads)


fn c_transform_simd_v2[
    N: Int
](mut state: QuantumState, control: Int, target: Int, gate: Gate):
    stride = 1 << target
    # In v2, we might dispatch to specialized kernels here
    c_transform_simd_base_v2[N](state, control, stride, gate)


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
        # transform_h(state, targets[j])
        transform_h_block_style(state, targets[j])
        for k in reversed(range(j)):
            c_transform_interval_p(
                state, targets[j], targets[k], -pi / 2 ** (j - k)
            )
            # c_transform_interval_p_precomputed(
            #     state, targets[j], targets[k], -pi / 2 ** (j - k)
            # )

    if swap:
        bit_reverse_state(state)


fn apply_cft_stage(
    mut state: QuantumState,
    target: Int,
    subspace_indices: List[Int],
    twiddle_stride: Int,
    ptr_fac_re: UnsafePointer[FloatType],
    ptr_fac_im: UnsafePointer[FloatType],
    inverse: Bool = False,
):
    """
    Apply a single DIF FFT stage to a target qubit.

    Args:
        state: The QuantumState to transform.
        target: The target qubit index.
        subspace_indices: Qubits in the transform that are logically LOWER than target.
        twiddle_stride: Stride multiplier for the factors table.
        ptr_fac_re: Precomputed real twiddle factors pointer.
        ptr_fac_im: Precomputed imaginary twiddle factors pointer.
        inverse: If True, applies the inverse stage.
    """
    var l = state.size()
    var stride = 1 << target
    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()

    # Scalar implementation for arbitrary bit configurations
    # This matches the logic of a single DIF stage.
    for k in range(l // (2 * stride)):
        var block_start = k * 2 * stride
        for idx in range(block_start, block_start + stride):
            var idx1 = idx + stride

            # Extract twiddle index j from the subspace bits
            var j = 0
            for i in range(len(subspace_indices)):
                if (idx >> subspace_indices[i]) & 1:
                    j |= 1 << i

            var tw_idx = j * twiddle_stride
            var w_re = ptr_fac_re[tw_idx]
            var w_im = ptr_fac_im[tw_idx]
            if inverse:
                w_im = -w_im

            var a_re = ptr_re[idx]
            var a_im = ptr_im[idx]
            var b_re = ptr_re[idx1]
            var b_im = ptr_im[idx1]

            var d_re = a_re - b_re
            var d_im = a_im - b_im

            ptr_re[idx] = a_re + b_re
            ptr_im[idx] = a_im + b_im
            ptr_re[idx1] = d_re * w_re - d_im * w_im
            ptr_im[idx1] = d_re * w_im + d_im * w_re


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


fn partial_bit_reverse_state(mut state: QuantumState, targets: List[Int]):
    """
    Perform bit-reversal swapping on a subset of qubits.
    """
    var k = len(targets)
    if k <= 1:
        return

    # Sort targets ascending
    var sorted_targets = targets.copy()
    for i in range(k):
        for j in range(i + 1, k):
            if sorted_targets[i] > sorted_targets[j]:
                var temp = sorted_targets[i]
                sorted_targets[i] = sorted_targets[j]
                sorted_targets[j] = temp

    var size = state.size()
    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()

    for i_large in range(size):
        # Extract small index i from large index i_large
        var i_small = 0
        for b in range(k):
            if (i_large >> sorted_targets[b]) & 1:
                i_small |= 1 << b

        # Reverse i_small to get j_small
        var j_small = 0
        for b in range(k):
            if (i_small >> b) & 1:
                j_small |= 1 << (k - 1 - b)

        if i_small < j_small:
            # Construct j_large from i_large by replacing target bits
            var j_large = i_large
            for b in range(k):
                var bit_val = (j_small >> b) & 1
                if bit_val:
                    j_large |= 1 << sorted_targets[b]
                else:
                    j_large &= ~(1 << sorted_targets[b])

            var tmp_re = ptr_re[i_large]
            var tmp_im = ptr_im[i_large]
            ptr_re[i_large] = ptr_re[j_large]
            ptr_im[i_large] = ptr_im[j_large]
            ptr_re[j_large] = tmp_re
            ptr_im[j_large] = tmp_im


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


@always_inline
fn compute_twiddle[
    inverse: Bool
](j: Int, n: Int) -> Tuple[FloatType, FloatType]:
    """Compute on-the-fly twiddle factor W_n^j."""
    alias angle_base = -2.0 * pi if not inverse else 2.0 * pi
    var angle = angle_base * Float64(j) / Float64(n)
    return cos(angle), sin(angle)


fn cft(
    mut state: QuantumState,
    targets: List[Int],
    inverse: Bool = False,
    do_swap: Bool = True,
):
    """
    Apply Classical Fourier Transform (CFT) to a contiguous set of qubits using the subspace approach.
    Assumes targets are sorted and contiguous: [b, b+1, ..., b+k-1].
    """
    var k = len(targets)
    if k == 0:
        return

    var n_total = state.size()
    var b = targets[0]
    var stride = 1 << b
    var span = 1 << k
    var num_subspaces = n_total // span

    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()
    alias scale = sqrt(0.5).cast[Type]()

    # Precompute Twiddle Table for k >= 4
    # Size: (span / 2) floats each for Re and Im
    var twiddle_storage_re = List[FloatType](capacity=1)
    var twiddle_storage_im = List[FloatType](capacity=1)
    twiddle_storage_re.append(0.0)
    twiddle_storage_im.append(0.0)
    var twiddle_ptr_re = twiddle_storage_re.unsafe_ptr()
    var twiddle_ptr_im = twiddle_storage_im.unsafe_ptr()

    if k >= 4:
        var half_span = span >> 1
        twiddle_storage_re = List[FloatType](capacity=half_span)
        twiddle_storage_im = List[FloatType](capacity=half_span)
        for _ in range(half_span):
            twiddle_storage_re.append(0.0)
            twiddle_storage_im.append(0.0)

        twiddle_ptr_re = twiddle_storage_re.unsafe_ptr()
        twiddle_ptr_im = twiddle_storage_im.unsafe_ptr()

        alias two_pi = 2.0 * pi
        var angle_base = two_pi if not inverse else -two_pi
        var base_n = Float64(span)

        for j in range(half_span):
            var angle = angle_base * Float64(j) / base_n
            twiddle_ptr_re[j] = cos(angle)
            twiddle_ptr_im[j] = sin(angle)

    @parameter
    fn subspace_worker(blk_idx: Int):
        # Calculate base offset using bit-insertion
        var low_mask = stride - 1
        var offset = (blk_idx & low_mask) | ((blk_idx & ~low_mask) << k)

        if k == 1:
            var i0 = offset
            var i1 = offset + stride
            var u_re = ptr_re[i0]
            var u_im = ptr_im[i0]
            var v_re = ptr_re[i1]
            var v_im = ptr_im[i1]
            ptr_re[i0] = (u_re + v_re) * scale
            ptr_im[i0] = (u_im + v_im) * scale
            ptr_re[i1] = (u_re - v_re) * scale
            ptr_im[i1] = (u_im - v_im) * scale
            return

        if k == 2:
            var i0 = offset
            var i1 = offset + stride
            var i2 = offset + 2 * stride
            var i3 = offset + 3 * stride
            var a0_re = ptr_re[i0]
            var a0_im = ptr_im[i0]
            var a1_re = ptr_re[i1]
            var a1_im = ptr_im[i1]
            var a2_re = ptr_re[i2]
            var a2_im = ptr_im[i2]
            var a3_re = ptr_re[i3]
            var a3_im = ptr_im[i3]

            if not inverse:
                if do_swap:
                    var t_re = a1_re
                    a1_re = a2_re
                    a2_re = t_re
                    var t_im = a1_im
                    a1_im = a2_im
                    a2_im = t_im
                var r0_re = (a0_re + a1_re) * scale
                var r0_im = (a0_im + a1_im) * scale
                var d0_re = (a0_re - a1_re) * scale
                var d0_im = (a0_im - a1_im) * scale
                var r1_re = (a2_re + a3_re) * scale
                var r1_im = (a2_im + a3_im) * scale
                var d1_re = (a2_re - a3_re) * scale
                var d1_im = (a2_im - a3_im) * scale
                ptr_re[i0] = (r0_re + r1_re) * scale
                ptr_im[i0] = (r0_im + r1_im) * scale
                ptr_re[i2] = (r0_re - r1_re) * scale
                ptr_im[i2] = (r0_im - r1_im) * scale
                var v1_rot_re = -d1_im
                var v1_rot_im = d1_re
                ptr_re[i1] = (d0_re + v1_rot_re) * scale
                ptr_im[i1] = (d0_im + v1_rot_im) * scale
                ptr_re[i3] = (d0_re - v1_rot_re) * scale
                ptr_im[i3] = (d0_im - v1_rot_im) * scale
            else:
                var r0_re = (a0_re + a2_re) * scale
                var r0_im = (a0_im + a2_im) * scale
                var d0_re = (a0_re - a2_re) * scale
                var d0_im = (a0_im - a2_im) * scale
                var r1_re = (a1_re + a3_re) * scale
                var r1_im = (a1_im + a3_im) * scale
                var d1_re = (a1_re - a3_re) * scale
                var d1_im = (a1_im - a3_im) * scale
                var v1_rot_re = d1_im
                var v1_rot_im = -d1_re  # W4^1_inv = -i
                var res0_re = (r0_re + r1_re) * scale
                var res0_im = (r0_im + r1_im) * scale
                var res1_re = (r0_re - r1_re) * scale
                var res1_im = (r0_im - r1_im) * scale
                var res2_re = (d0_re + v1_rot_re) * scale
                var res2_im = (d0_im + v1_rot_im) * scale
                var res3_re = (d0_re - v1_rot_re) * scale
                var res3_im = (d0_im - v1_rot_im) * scale
                if do_swap:
                    ptr_re[i0] = res0_re
                    ptr_im[i0] = res0_im
                    ptr_re[i1] = res2_re
                    ptr_im[i1] = res2_im
                    ptr_re[i2] = res1_re
                    ptr_im[i2] = res1_im
                    ptr_re[i3] = res3_re
                    ptr_im[i3] = res3_im
                else:
                    ptr_re[i0] = res0_re
                    ptr_im[i0] = res0_im
                    ptr_re[i1] = res1_re
                    ptr_im[i1] = res1_im
                    ptr_re[i2] = res2_re
                    ptr_im[i2] = res2_im
                    ptr_re[i3] = res3_re
                    ptr_im[i3] = res3_im
            return

        if k == 3:
            var i0 = offset
            var i1 = offset + stride
            var i2 = offset + 2 * stride
            var i3 = offset + 3 * stride
            var i4 = offset + 4 * stride
            var i5 = offset + 5 * stride
            var i6 = offset + 6 * stride
            var i7 = offset + 7 * stride
            var a0_re = ptr_re[i0]
            var a0_im = ptr_im[i0]
            var a1_re = ptr_re[i1]
            var a1_im = ptr_im[i1]
            var a2_re = ptr_re[i2]
            var a2_im = ptr_im[i2]
            var a3_re = ptr_re[i3]
            var a3_im = ptr_im[i3]
            var a4_re = ptr_re[i4]
            var a4_im = ptr_im[i4]
            var a5_re = ptr_re[i5]
            var a5_im = ptr_im[i5]
            var a6_re = ptr_re[i6]
            var a6_im = ptr_im[i6]
            var a7_re = ptr_re[i7]
            var a7_im = ptr_im[i7]

            if not inverse:
                if do_swap:
                    var t_re = a1_re
                    a1_re = a4_re
                    a4_re = t_re
                    var t_im = a1_im
                    a1_im = a4_im
                    a4_im = t_im
                    t_re = a3_re
                    a3_re = a6_re
                    a6_re = t_re
                    t_im = a3_im
                    a3_im = a6_im
                    a6_im = t_im
                var s0r0_re = (a0_re + a1_re) * scale
                var s0r0_im = (a0_im + a1_im) * scale
                var s0d0_re = (a0_re - a1_re) * scale
                var s0d0_im = (a0_im - a1_im) * scale
                var s0r1_re = (a2_re + a3_re) * scale
                var s0r1_im = (a2_im + a3_im) * scale
                var s0d1_re = (a2_re - a3_re) * scale
                var s0d1_im = (a2_im - a3_im) * scale
                var s0r2_re = (a4_re + a5_re) * scale
                var s0r2_im = (a4_im + a5_im) * scale
                var s0d2_re = (a4_re - a5_re) * scale
                var s0d2_im = (a4_im - a5_im) * scale
                var s0r3_re = (a6_re + a7_re) * scale
                var s0r3_im = (a6_im + a7_im) * scale
                var s0d3_re = (a6_re - a7_re) * scale
                var s0d3_im = (a6_im - a7_im) * scale
                var s1r0_re = (s0r0_re + s0r1_re) * scale
                var s1r0_im = (s0r0_im + s0r1_im) * scale
                var s1d0_re = (s0r0_re - s0r1_re) * scale
                var s1d0_im = (s0r0_im - s0r1_im) * scale
                var v1_rot_re = -s0d1_im
                var v1_rot_im = s0d1_re
                var s1r1_re = (s0d0_re + v1_rot_re) * scale
                var s1r1_im = (s0d0_im + v1_rot_im) * scale
                var s1d1_re = (s0d0_re - v1_rot_re) * scale
                var s1d1_im = (s0d0_im - v1_rot_im) * scale
                var s1r2_re = (s0r2_re + s0r3_re) * scale
                var s1r2_im = (s0r2_im + s0r3_im) * scale
                var s1d2_re = (s0r2_re - s0r3_re) * scale
                var s1d2_im = (s0r2_im - s0r3_im) * scale
                var v3_rot_re = -s0d3_im
                var v3_rot_im = s0d3_re
                var s1r3_re = (s0d2_re + v3_rot_re) * scale
                var s1r3_im = (s0d2_im + v3_rot_im) * scale
                var s1d3_re = (s0d2_re - v3_rot_re) * scale
                var s1d3_im = (s0d2_im - v3_rot_im) * scale
                ptr_re[i0] = (s1r0_re + s1r2_re) * scale
                ptr_im[i0] = (s1r0_im + s1r2_im) * scale
                ptr_re[i4] = (s1r0_re - s1r2_re) * scale
                ptr_im[i4] = (s1r0_im - s1r2_im) * scale
                var v5_rot_re = (s1r3_re - s1r3_im) * scale
                var v5_rot_im = (s1r3_re + s1r3_im) * scale
                ptr_re[i1] = (s1r1_re + v5_rot_re) * scale
                ptr_im[i1] = (s1r1_im + v5_rot_im) * scale
                ptr_re[i5] = (s1r1_re - v5_rot_re) * scale
                ptr_im[i5] = (s1r1_im - v5_rot_im) * scale
                var v6_rot_re = -s1d2_im
                var v6_rot_im = s1d2_re
                ptr_re[i2] = (s1d0_re + v6_rot_re) * scale
                ptr_im[i2] = (s1d0_im + v6_rot_im) * scale
                ptr_re[i6] = (s1d0_re - v6_rot_re) * scale
                ptr_im[i6] = (s1d0_im - v6_rot_im) * scale
                var v7_rot_re = (-s1d3_re - s1d3_im) * scale
                var v7_rot_im = (s1d3_re - s1d3_im) * scale
                ptr_re[i3] = (s1d1_re + v7_rot_re) * scale
                ptr_im[i3] = (s1d1_im + v7_rot_im) * scale
                ptr_re[i7] = (s1d1_re - v7_rot_re) * scale
                ptr_im[i7] = (s1d1_im - v7_rot_im) * scale
            else:
                var s2r0_re = (a0_re + a4_re) * scale
                var s2r0_im = (a0_im + a4_im) * scale
                var s2d0_re = (a0_re - a4_re) * scale
                var s2d0_im = (a0_im - a4_im) * scale
                var s2r1_re = (a1_re + a5_re) * scale
                var s2r1_im = (a1_im + a5_im) * scale
                var s2d1_re = (a1_re - a5_re) * scale
                var s2d1_im = (a1_im - a5_im) * scale
                var s2r2_re = (a2_re + a6_re) * scale
                var s2r2_im = (a2_im + a6_im) * scale
                var s2d2_re = (a2_re - a6_re) * scale
                var s2d2_im = (a2_im - a6_im) * scale
                var s2r3_re = (a3_re + a7_re) * scale
                var s2r3_im = (a3_im + a7_im) * scale
                var s2d3_re = (a3_re - a7_re) * scale
                var s2d3_im = (a3_im - a7_im) * scale
                var s2d1_rot_re = (s2d1_re + s2d1_im) * scale
                var s2d1_rot_im = (s2d1_im - s2d1_re) * scale
                var s2d2_rot_re = s2d2_im
                var s2d2_rot_im = -s2d2_re
                var s2d3_rot_re = (s2d3_im - s2d3_re) * scale
                var s2d3_rot_im = (-s2d3_re - s2d3_im) * scale
                var s1r0_re = (s2r0_re + s2r2_re) * scale
                var s1r0_im = (s2r0_im + s2r2_im) * scale
                var s1d0_re = (s2r0_re - s2r2_re) * scale
                var s1d0_im = (s2r0_im - s2r2_im) * scale
                var s1r1_re = (s2r1_re + s2r3_re) * scale
                var s1r1_im = (s2r1_im + s2r3_im) * scale
                var s1d1_re = (s2r1_re - s2r3_re) * scale
                var s1d1_im = (s2r1_im - s2r3_im) * scale
                var s1r2_re = (s2d0_re + s2d2_rot_re) * scale
                var s1r2_im = (s2d0_im + s2d2_rot_im) * scale
                var s1d2_re = (s2d0_re - s2d2_rot_re) * scale
                var s1d2_im = (s2d0_im - s2d2_rot_im) * scale
                var s1r3_re = (s2d1_rot_re + s2d3_rot_re) * scale
                var s1r3_im = (s2d1_rot_im + s2d3_rot_im) * scale
                var s1d3_re = (s2d1_rot_re - s2d3_rot_re) * scale
                var s1d3_im = (s2d1_rot_im - s2d3_rot_im) * scale
                var s1d1_rot_re = s1d1_im
                var s1d1_rot_im = -s1d1_re
                var s1d3_rot_re = s1d3_im
                var s1d3_rot_im = -s1d3_re
                var res0_re = (s1r0_re + s1r1_re) * scale
                var res0_im = (s1r0_im + s1r1_im) * scale
                var res1_re = (s1r0_re - s1r1_re) * scale
                var res1_im = (s1r0_im - s1r1_im) * scale
                var res2_re = (s1d0_re + s1d1_rot_re) * scale
                var res2_im = (s1d0_im + s1d1_rot_im) * scale
                var res3_re = (s1d0_re - s1d1_rot_re) * scale
                var res3_im = (s1d0_im - s1d1_rot_im) * scale
                var res4_re = (s1r2_re + s1r3_re) * scale
                var res4_im = (s1r2_im + s1r3_im) * scale
                var res5_re = (s1r2_re - s1r3_re) * scale
                var res5_im = (s1r2_im - s1r3_im) * scale
                var res6_re = (s1d2_re + s1d3_rot_re) * scale
                var res6_im = (s1d2_im + s1d3_rot_im) * scale
                var res7_re = (s1d2_re - s1d3_rot_re) * scale
                var res7_im = (s1d2_im - s1d3_rot_im) * scale
                if do_swap:
                    ptr_re[i0] = res0_re
                    ptr_im[i0] = res0_im
                    ptr_re[i1] = res4_re
                    ptr_im[i1] = res4_im
                    ptr_re[i2] = res2_re
                    ptr_im[i2] = res2_im
                    ptr_re[i3] = res6_re
                    ptr_im[i3] = res6_im
                    ptr_re[i4] = res1_re
                    ptr_im[i4] = res1_im
                    ptr_re[i5] = res5_re
                    ptr_im[i5] = res5_im
                    ptr_re[i6] = res3_re
                    ptr_im[i6] = res3_im
                    ptr_re[i7] = res7_re
                    ptr_im[i7] = res7_im
                else:
                    ptr_re[i0] = res0_re
                    ptr_im[i0] = res0_im
                    ptr_re[i1] = res1_re
                    ptr_im[i1] = res1_im
                    ptr_re[i2] = res2_re
                    ptr_im[i2] = res2_im
                    ptr_re[i3] = res3_re
                    ptr_im[i3] = res3_im
                    ptr_re[i4] = res4_re
                    ptr_im[i4] = res4_im
                    ptr_re[i5] = res5_re
                    ptr_im[i5] = res5_im
                    ptr_re[i6] = res6_re
                    ptr_im[i6] = res6_im
                    ptr_re[i7] = res7_re
                    ptr_im[i7] = res7_im
            return

        if not inverse:
            # Forward QFT: Swap(start) then DIT(0..k-1)
            # 1. Local Swap at START
            if do_swap:
                for i in range(span):
                    var reversed_i = 0
                    for b_idx in range(k):
                        if (i >> b_idx) & 1:
                            reversed_i |= 1 << (k - 1 - b_idx)

                    if i < reversed_i:
                        var idx_i = offset + i * stride
                        var idx_r = offset + reversed_i * stride
                        var tmp_re = ptr_re[idx_i]
                        var tmp_im = ptr_im[idx_i]
                        ptr_re[idx_i] = ptr_re[idx_r]
                        ptr_im[idx_i] = ptr_im[idx_r]
                        ptr_re[idx_r] = tmp_re
                        ptr_im[idx_r] = tmp_im


fn dagger(u: List[Amplitude], m: Int) -> List[Amplitude]:
    """Computes the conjugate transpose (dagger) of a 2^m x 2^m matrix.

    Args:
        u: Flattened unitary matrix.
        m: Number of qubits the unitary acts on (dimension is 2^m x 2^m).

    Returns:
        The conjugate transpose of u.
    """
    var dim = 1 << m
    var res = List[Amplitude](capacity=len(u))
    # dagger(U)_{ij} = conj(U_{ji})
    for i in range(dim):
        for j in range(dim):
            var val = u[j * dim + i]
            res.append(Amplitude(val.re, -val.im))
    return res^


fn dagger(g: Gate) -> Gate:
    """Computes the conjugate transpose (dagger) of a 2x2 gate matrix."""
    return Gate(
        InlineArray[Amplitude, 2](
            Amplitude(g[0][0].re, -g[0][0].im),
            Amplitude(g[1][0].re, -g[1][0].im),
        ),
        InlineArray[Amplitude, 2](
            Amplitude(g[0][1].re, -g[0][1].im),
            Amplitude(g[1][1].re, -g[1][1].im),
        ),
    )


fn transform_u(mut state: QuantumState, U: List[Amplitude], t: Int, m: Int):
    """
    Applies an arbitrary unitary matrix U (of size 2^m x 2^m)
    starting at target qubit t.
    """
    var n = Int(log2(Float64(state.size())))
    var vec_size = 1 << m

    for suffix in range(1 << t):
        for prefix in range(1 << (n - m - t)):
            var vec = List[Amplitude](capacity=vec_size)
            for target in range(vec_size):
                var k = (prefix << (t + m)) + (target << t) + suffix
                vec.append(state[k])

            var vec_out = mat_vec_mul(U, vec)

            for target in range(vec_size):
                var k = (prefix << (t + m)) + (target << t) + suffix
                state[k] = vec_out[target]


fn c_transform_u(
    mut state: QuantumState, U: List[Amplitude], c: Int, t: Int, m: Int
):
    """
    Applies an arbitrary unitary matrix U (of size 2^m x 2^m)
    starting at target qubit t, controlled by qubit c.
    """
    var n = Int(log2(Float64(state.size())))
    var vec_size = 1 << m

    for suffix in range(1 << t):
        for prefix in range(1 << (n - m - t)):
            var vec = List[Amplitude](capacity=vec_size)
            for idx in range(vec_size):
                var k = (prefix << (t + m)) + (idx << t) + suffix
                if is_bit_set(k, c):
                    vec.append(state[k])
                else:
                    vec.append(Amplitude(0.0))

            var vec_out = mat_vec_mul(U, vec)

            for idx in range(vec_size):
                var k = (prefix << (t + m)) + (idx << t) + suffix
                if is_bit_set(k, c):
                    state[k] = vec_out[idx]
