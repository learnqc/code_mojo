from butterfly.core.state import QuantumState
from butterfly.core.types import Gate, FloatType, sq_half_re
from butterfly.core.gates import GateKind
from butterfly.utils.context import ExecContext
from algorithm import parallelize
from math import cos, sin


# @always_inline
fn process_pair_scalar(mut state: QuantumState, gate: Gate, k0: Int, k1: Int):
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


@always_inline
fn is_bit_set(m: Int, k: Int) -> Bool:
    return m & Int(1 << k) != 0


@always_inline
fn block_bounds(
    total_blocks: Int,
    workers: Int,
    blocks_per_worker: Int,
    item_id: Int,
) -> Tuple[Int, Int]:
    var start_block = item_id * blocks_per_worker
    var end_block = (
        total_blocks if item_id
        == workers - 1 else (item_id + 1) * blocks_per_worker
    )
    return (start_block, end_block)


@always_inline
fn transform_scalar(
    mut state: QuantumState,
    target: Int,
    gate: Gate,
    gate_kind: Int = -1,
    gate_arg: FloatType = 0,
    ctx: ExecContext = ExecContext(),
):
    """Scalar single-qubit transform (sequential, no runtime branching)."""
    if gate_kind == GateKind.H and ctx.simd_use_specialized_h:
        transform_h_scalar(state, target, ctx)
        return
    if gate_kind == GateKind.P and ctx.simd_use_specialized_p:
        transform_p_scalar(state, target, gate_arg, ctx)
        return
    if gate_kind == GateKind.X and ctx.simd_use_specialized_x:
        transform_x_scalar(state, target, ctx)
        return
    if gate_kind == GateKind.RY and ctx.simd_use_specialized_ry:
        transform_ry_scalar(state, target, gate_arg, ctx)
        return

    var l = state.size()
    var stride = 1 << target
    var threads = ctx.threads
    if threads > 0:
        var total_blocks = l // (2 * stride)
        if total_blocks == 0:
            return
        var workers = min(threads, total_blocks)
        var blocks_per_worker = max(1, total_blocks // workers)

        @parameter
        fn worker(item_id: Int):
            var (start_block, end_block) = block_bounds(
                total_blocks, workers, blocks_per_worker, item_id
            )
            for b in range(start_block, end_block):
                var k = b * 2 * stride
                for m in range(stride):
                    process_pair_scalar(state, gate, k + m, k + m + stride)

        if workers > 1:
            parallelize[worker](workers, threads)
        else:
            worker(0)
    else:
        var r = 0
        for j in range(l // 2):
            var idx = 2 * j - r
            process_pair_scalar(state, gate, idx, idx + stride)
            r += 1
            if r == stride:
                r = 0


@always_inline
fn c_transform_scalar(
    mut state: QuantumState,
    control: Int,
    target: Int,
    gate: Gate,
    gate_kind: Int = -1,
    gate_arg: FloatType = 0,
    ctx: ExecContext = ExecContext(),
):
    """Scalar controlled single-qubit transform (sequential)."""
    if gate_kind == GateKind.P and ctx.simd_use_specialized_cp:
        c_transform_p_scalar(state, control, target, gate_arg, ctx)
        return
    if gate_kind == GateKind.X and ctx.simd_use_specialized_cx:
        c_transform_x_scalar(state, control, target, ctx)
        return
    if gate_kind == GateKind.RY and ctx.simd_use_specialized_cry:
        c_transform_ry_scalar(state, control, target, gate_arg, ctx)
        return

    var stride = 1 << target
    var l = state.size()
    var threads = ctx.threads
    if threads > 0:
        var total_blocks = l // (2 * stride)
        if total_blocks == 0:
            return
        var workers = min(threads, total_blocks)
        var blocks_per_worker = max(1, total_blocks // workers)

        @parameter
        fn worker(item_id: Int):
            var (start_block, end_block) = block_bounds(
                total_blocks, workers, blocks_per_worker, item_id
            )
            for b in range(start_block, end_block):
                var k = b * 2 * stride
                for m in range(stride):
                    var idx = k + m
                    if is_bit_set(idx, control):
                        process_pair_scalar(state, gate, idx, idx + stride)

        if workers > 1:
            parallelize[worker](workers, threads)
        else:
            worker(0)
    else:
        var r = 0
        for j in range(state.size() // 2):
            var idx = 2 * j - r
            if is_bit_set(idx, control):
                process_pair_scalar(state, gate, idx, idx + stride)
            r += 1
            if r == stride:
                r = 0


fn transform_x_scalar(
    mut state: QuantumState,
    target: Int,
    ctx: ExecContext = ExecContext(),
):
    """Scalar X transform (swap amplitudes)."""
    var l = state.size()
    var stride = 1 << target
    var threads = ctx.threads
    if threads > 0:
        var total_blocks = l // (2 * stride)
        if total_blocks == 0:
            return
        var workers = min(threads, total_blocks)
        var blocks_per_worker = max(1, total_blocks // workers)

        @parameter
        fn worker(item_id: Int):
            var (start_block, end_block) = block_bounds(
                total_blocks, workers, blocks_per_worker, item_id
            )
            for b in range(start_block, end_block):
                var k = b * 2 * stride
                for m in range(stride):
                    var idx = k + m
                    var u_re = state.re[idx]
                    var u_im = state.im[idx]
                    var v_re = state.re[idx + stride]
                    var v_im = state.im[idx + stride]
                    state.re[idx] = v_re
                    state.im[idx] = v_im
                    state.re[idx + stride] = u_re
                    state.im[idx + stride] = u_im

        if workers > 1:
            parallelize[worker](workers, threads)
        else:
            worker(0)
    else:
        var r = 0
        for j in range(l // 2):
            var idx = 2 * j - r
            var u_re = state.re[idx]
            var u_im = state.im[idx]
            var v_re = state.re[idx + stride]
            var v_im = state.im[idx + stride]
            state.re[idx] = v_re
            state.im[idx] = v_im
            state.re[idx + stride] = u_re
            state.im[idx + stride] = u_im
            r += 1
            if r == stride:
                r = 0


fn transform_h_scalar(
    mut state: QuantumState,
    target: Int,
    ctx: ExecContext = ExecContext(),
):
    """Scalar Hadamard transform."""
    var l = state.size()
    var stride = 1 << target
    alias sq_half = 0.7071067811865476
    var threads = ctx.threads
    if threads > 0:
        var total_blocks = l // (2 * stride)
        if total_blocks == 0:
            return
        var workers = min(threads, total_blocks)
        var blocks_per_worker = max(1, total_blocks // workers)

        @parameter
        fn worker(item_id: Int):
            var (start_block, end_block) = block_bounds(
                total_blocks, workers, blocks_per_worker, item_id
            )
            for b in range(start_block, end_block):
                var k = b * 2 * stride
                for m in range(stride):
                    var idx = k + m
                    var u_re = state.re[idx]
                    var u_im = state.im[idx]
                    var v_re = state.re[idx + stride]
                    var v_im = state.im[idx + stride]
                    state.re[idx] = (u_re + v_re) * sq_half
                    state.im[idx] = (u_im + v_im) * sq_half
                    state.re[idx + stride] = (u_re - v_re) * sq_half
                    state.im[idx + stride] = (u_im - v_im) * sq_half

        if workers > 1:
            parallelize[worker](workers, threads)
        else:
            worker(0)
    else:
        var r = 0
        for j in range(l // 2):
            var idx = 2 * j - r
            var u_re = state.re[idx]
            var u_im = state.im[idx]
            var v_re = state.re[idx + stride]
            var v_im = state.im[idx + stride]
            state.re[idx] = (u_re + v_re) * sq_half
            state.im[idx] = (u_im + v_im) * sq_half
            state.re[idx + stride] = (u_re - v_re) * sq_half
            state.im[idx + stride] = (u_im - v_im) * sq_half
            r += 1
            if r == stride:
                r = 0


fn transform_p_scalar(
    mut state: QuantumState,
    target: Int,
    theta: FloatType,
    ctx: ExecContext = ExecContext(),
):
    """Scalar Phase transform."""
    var l = state.size()
    var stride = 1 << target
    var cos_t = cos(Float64(theta))
    var sin_t = sin(Float64(theta))
    var threads = ctx.threads
    if threads > 0:
        var total_blocks = l // (2 * stride)
        if total_blocks == 0:
            return
        var workers = min(threads, total_blocks)
        var blocks_per_worker = max(1, total_blocks // workers)

        @parameter
        fn worker(item_id: Int):
            var (start_block, end_block) = block_bounds(
                total_blocks, workers, blocks_per_worker, item_id
            )
            for b in range(start_block, end_block):
                var k = b * 2 * stride
                for m in range(stride):
                    var idx = k + m + stride
                    var v_re = state.re[idx]
                    var v_im = state.im[idx]
                    state.re[idx] = v_re * cos_t - v_im * sin_t
                    state.im[idx] = v_re * sin_t + v_im * cos_t

        if workers > 1:
            parallelize[worker](workers, threads)
        else:
            worker(0)
    else:
        var r = 0
        for j in range(l // 2):
            var idx = 2 * j - r + stride
            var v_re = state.re[idx]
            var v_im = state.im[idx]
            state.re[idx] = v_re * cos_t - v_im * sin_t
            state.im[idx] = v_re * sin_t + v_im * cos_t
            r += 1
            if r == stride:
                r = 0


fn transform_ry_scalar(
    mut state: QuantumState,
    target: Int,
    theta: FloatType,
    ctx: ExecContext = ExecContext(),
):
    """Scalar RY transform."""
    var l = state.size()
    var stride = 1 << target
    var cos_t = cos(Float64(theta) / 2)
    var sin_t = sin(Float64(theta) / 2)
    var threads = ctx.threads
    if threads > 0:
        var total_blocks = l // (2 * stride)
        if total_blocks == 0:
            return
        var workers = min(threads, total_blocks)
        var blocks_per_worker = max(1, total_blocks // workers)

        @parameter
        fn worker(item_id: Int):
            var (start_block, end_block) = block_bounds(
                total_blocks, workers, blocks_per_worker, item_id
            )
            for b in range(start_block, end_block):
                var k = b * 2 * stride
                for m in range(stride):
                    var idx = k + m
                    var u_re = state.re[idx]
                    var u_im = state.im[idx]
                    var v_re = state.re[idx + stride]
                    var v_im = state.im[idx + stride]
                    state.re[idx] = u_re * cos_t - v_re * sin_t
                    state.im[idx] = u_im * cos_t - v_im * sin_t
                    state.re[idx + stride] = u_re * sin_t + v_re * cos_t
                    state.im[idx + stride] = u_im * sin_t + v_im * cos_t

        if workers > 1:
            parallelize[worker](workers, threads)
        else:
            worker(0)
    else:
        var r = 0
        for j in range(l // 2):
            var idx = 2 * j - r
            var u_re = state.re[idx]
            var u_im = state.im[idx]
            var v_re = state.re[idx + stride]
            var v_im = state.im[idx + stride]
            state.re[idx] = u_re * cos_t - v_re * sin_t
            state.im[idx] = u_im * cos_t - v_im * sin_t
            state.re[idx + stride] = u_re * sin_t + v_re * cos_t
            state.im[idx + stride] = u_im * sin_t + v_im * cos_t
            r += 1
            if r == stride:
                r = 0


fn c_transform_x_scalar(
    mut state: QuantumState,
    control: Int,
    target: Int,
    ctx: ExecContext = ExecContext(),
):
    """Scalar controlled X transform."""
    var stride = 1 << target
    var l = state.size()
    var threads = ctx.threads
    if threads > 0:
        var total_blocks = l // (2 * stride)
        if total_blocks == 0:
            return
        var workers = min(threads, total_blocks)
        var blocks_per_worker = max(1, total_blocks // workers)

        @parameter
        fn worker(item_id: Int):
            var (start_block, end_block) = block_bounds(
                total_blocks, workers, blocks_per_worker, item_id
            )
            for b in range(start_block, end_block):
                var k = b * 2 * stride
                for m in range(stride):
                    var idx = k + m
                    if is_bit_set(idx, control):
                        var u_re = state.re[idx]
                        var u_im = state.im[idx]
                        var v_re = state.re[idx + stride]
                        var v_im = state.im[idx + stride]
                        state.re[idx] = v_re
                        state.im[idx] = v_im
                        state.re[idx + stride] = u_re
                        state.im[idx + stride] = u_im

        if workers > 1:
            parallelize[worker](workers, threads)
        else:
            worker(0)
    else:
        var r = 0
        for j in range(state.size() // 2):
            var idx = 2 * j - r
            if is_bit_set(idx, control):
                var u_re = state.re[idx]
                var u_im = state.im[idx]
                var v_re = state.re[idx + stride]
                var v_im = state.im[idx + stride]
                state.re[idx] = v_re
                state.im[idx] = v_im
                state.re[idx + stride] = u_re
                state.im[idx + stride] = u_im
            r += 1
            if r == stride:
                r = 0


fn c_transform_p_scalar(
    mut state: QuantumState,
    control: Int,
    target: Int,
    theta: FloatType,
    ctx: ExecContext = ExecContext(),
):
    """Scalar controlled Phase transform."""
    var stride = 1 << target
    var cos_t = cos(Float64(theta))
    var sin_t = sin(Float64(theta))
    var l = state.size()
    var threads = ctx.threads
    if threads > 0:
        var total_blocks = l // (2 * stride)
        if total_blocks == 0:
            return
        var workers = min(threads, total_blocks)
        var blocks_per_worker = max(1, total_blocks // workers)

        @parameter
        fn worker(item_id: Int):
            var (start_block, end_block) = block_bounds(
                total_blocks, workers, blocks_per_worker, item_id
            )
            for b in range(start_block, end_block):
                var k = b * 2 * stride
                for m in range(stride):
                    var idx = k + m
                    if is_bit_set(idx, control):
                        var v_re = state.re[idx + stride]
                        var v_im = state.im[idx + stride]
                        state.re[idx + stride] = v_re * cos_t - v_im * sin_t
                        state.im[idx + stride] = v_re * sin_t + v_im * cos_t

        if workers > 1:
            parallelize[worker](workers, threads)
        else:
            worker(0)
    else:
        var r = 0
        for j in range(state.size() // 2):
            var idx = 2 * j - r
            if is_bit_set(idx, control):
                var v_re = state.re[idx + stride]
                var v_im = state.im[idx + stride]
                state.re[idx + stride] = v_re * cos_t - v_im * sin_t
                state.im[idx + stride] = v_re * sin_t + v_im * cos_t
            r += 1
            if r == stride:
                r = 0


fn c_transform_ry_scalar(
    mut state: QuantumState,
    control: Int,
    target: Int,
    theta: FloatType,
    ctx: ExecContext = ExecContext(),
):
    """Scalar controlled RY transform."""
    var stride = 1 << target
    var cos_t = cos(Float64(theta) / 2)
    var sin_t = sin(Float64(theta) / 2)
    var l = state.size()
    var threads = ctx.threads
    if threads > 0:
        var total_blocks = l // (2 * stride)
        if total_blocks == 0:
            return
        var workers = min(threads, total_blocks)
        var blocks_per_worker = max(1, total_blocks // workers)

        @parameter
        fn worker(item_id: Int):
            var (start_block, end_block) = block_bounds(
                total_blocks, workers, blocks_per_worker, item_id
            )
            for b in range(start_block, end_block):
                var k = b * 2 * stride
                for m in range(stride):
                    var idx = k + m
                    if is_bit_set(idx, control):
                        var u_re = state.re[idx]
                        var u_im = state.im[idx]
                        var v_re = state.re[idx + stride]
                        var v_im = state.im[idx + stride]
                        state.re[idx] = u_re * cos_t - v_re * sin_t
                        state.im[idx] = u_im * cos_t - v_im * sin_t
                        state.re[idx + stride] = u_re * sin_t + v_re * cos_t
                        state.im[idx + stride] = u_im * sin_t + v_im * cos_t

        if workers > 1:
            parallelize[worker](workers, threads)
        else:
            worker(0)
    else:
        var r = 0
        for j in range(state.size() // 2):
            var idx = 2 * j - r
            if is_bit_set(idx, control):
                var u_re = state.re[idx]
                var u_im = state.im[idx]
                var v_re = state.re[idx + stride]
                var v_im = state.im[idx + stride]
                state.re[idx] = u_re * cos_t - v_re * sin_t
                state.im[idx] = u_im * cos_t - v_im * sin_t
                state.re[idx + stride] = u_re * sin_t + v_re * cos_t
                state.im[idx + stride] = u_im * sin_t + v_im * cos_t
            r += 1
            if r == stride:
                r = 0


fn mc_transform_scalar(
    mut state: QuantumState,
    controls: List[Int],
    target: Int,
    gate: Gate,
    ctx: ExecContext = ExecContext(),
):
    """Scalar generic multi-controlled single-qubit gate transform."""
    _ = ctx
    if len(controls) == 0:
        transform_scalar(state, target, gate, ctx=ctx)
        return

    var mask = 0
    for c in controls:
        mask |= 1 << c

    var stride = 1 << target
    var r = 0
    for j in range(state.size() // 2):
        var idx = 2 * j - r
        if (idx & mask) == mask:
            process_pair_scalar(state, gate, idx, idx + stride)
        r += 1
        if r == stride:
            r = 0


fn mc_transform_scalar_block(
    mut state: QuantumState,
    controls: List[Int],
    target: Int,
    gate: Gate,
    ctx: ExecContext = ExecContext(),
):
    """Scalar multi-controlled gate using block enumeration (no per-index checks)."""
    _ = ctx
    if len(controls) == 0:
        transform_scalar(state, target, gate, ctx=ctx)
        return

    var size = state.size()
    var n = 0
    var tmp = size
    while tmp > 1:
        tmp //= 2
        n += 1

    var mask = 0
    for c in controls:
        mask |= 1 << c

    var stride = 1 << target
    var free_bits = List[Int]()
    for b in range(n):
        if b == target:
            continue
        if (mask & (1 << b)) != 0:
            continue
        free_bits.append(b)

    var free_count = len(free_bits)
    var combos = 1 << free_count
    for assignment in range(combos):
        var idx0 = mask
        for i in range(free_count):
            if (assignment >> i) & 1 != 0:
                idx0 |= 1 << free_bits[i]
        process_pair_scalar(state, gate, idx0, idx0 + stride)

@always_inline
fn process_h_pair_scalar(
    ptr_re: UnsafePointer[FloatType, MutAnyOrigin],
    ptr_im: UnsafePointer[FloatType, MutAnyOrigin],
    idx: Int,
    stride: Int,
):
    var u_re = ptr_re[idx]
    var u_im = ptr_im[idx]
    var v_re = ptr_re[idx + stride]
    var v_im = ptr_im[idx + stride]
    ptr_re[idx] = (u_re + v_re) * sq_half_re
    ptr_im[idx] = (u_im + v_im) * sq_half_re
    ptr_re[idx + stride] = (u_re - v_re) * sq_half_re
    ptr_im[idx + stride] = (u_im - v_im) * sq_half_re


fn process_p_pair_scalar(
    ptr_re: UnsafePointer[FloatType, MutAnyOrigin],
    ptr_im: UnsafePointer[FloatType, MutAnyOrigin],
    idx: Int,
    cos_t: Float64,
    sin_t: Float64,
):
    var v_re = ptr_re[idx]
    var v_im = ptr_im[idx]
    ptr_re[idx] = v_re * cos_t - v_im * sin_t
    ptr_im[idx] = v_im * cos_t + v_re * sin_t
