from butterfly.core.state import QuantumState, bit_reverse_state
from butterfly.core.types import FloatType, Gate, simd_width, sq_half_re
from butterfly.core.gates import H, P
from butterfly.core.transformations_scalar import process_h_pair_scalar, process_p_pair_scalar
from butterfly.utils.common import detect_logical_cores
from butterfly.utils.context import ExecContext
from collections import List

from algorithm import parallelize, vectorize
from math import cos, sin, pi


fn get_num_chunks(ctx: ExecContext) -> Int:
    """Return chunk count for SIMD/parallel ref kernels (fallback: cores)."""
    var configured = ctx.quantum_simd_parallel_chunks
    if configured > 0:
        return configured
    # Match simd_v2 default chunking.
    return 8


fn get_threads(ctx: ExecContext) -> Int:
    """Return thread override from config, or 0 for runtime default."""
    var configured = ctx.threads
    return configured if configured > 0 else 0


@always_inline
fn block_bounds(
    total_blocks: Int, actual_workers: Int, blocks_per_worker: Int, item_id: Int
) -> Tuple[Int, Int]:
    var start_block = item_id * blocks_per_worker
    var end_block = (
        total_blocks if item_id
        == actual_workers - 1 else (item_id + 1) * blocks_per_worker
    )
    return (start_block, end_block)


@always_inline
fn transform_h_simd_parallel(
    mut state: QuantumState,
    target: Int,
    ctx: ExecContext = ExecContext(),
):
    """Parallelized vectorized Hadamard gate (ref baseline)."""
    var l = state.size()
    var stride = 1 << target
    var ptr_re = state.re_ptr()
    var ptr_im = state.im_ptr()

    var num_work_items = get_num_chunks(ctx)
    var total_blocks = l // (2 * stride)
    if total_blocks == 0:
        return
    var actual_workers = min(num_work_items, total_blocks)
    var blocks_per_worker = max(1, total_blocks // num_work_items)
    var threads = get_threads(ctx)

    @always_inline
    fn process_block(k: Int):
        @parameter
        fn vectorize_h[width: Int](m: Int):
            var idx = k + m
            var u_re = ptr_re.load[width=width](idx)
            var u_im = ptr_im.load[width=width](idx)
            var v_re = ptr_re.load[width=width](idx + stride)
            var v_im = ptr_im.load[width=width](idx + stride)

            var sum_re = (u_re + v_re) * sq_half_re
            var sum_im = (u_im + v_im) * sq_half_re
            var diff_re = (u_re - v_re) * sq_half_re
            var diff_im = (u_im - v_im) * sq_half_re

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
                ptr_re[idx] = (u_re + v_re) * sq_half_re
                ptr_im[idx] = (u_im + v_im) * sq_half_re
                ptr_re[idx + stride] = (u_re - v_re) * sq_half_re
                ptr_im[idx + stride] = (u_im - v_im) * sq_half_re

    @parameter
    fn worker(item_id: Int):
        var start_block = item_id * blocks_per_worker
        var end_block = (
            total_blocks if item_id
            == actual_workers - 1 else (item_id + 1) * blocks_per_worker
        )
        for k_idx in range(start_block, end_block):
            process_block(k_idx * 2 * stride)

    if actual_workers > 1:
        if threads > 0:
            parallelize[worker](actual_workers, threads)
        else:
            parallelize[worker](actual_workers)
    else:
        worker(0)


@always_inline
fn transform_x_simd_parallel(
    mut state: QuantumState,
    target: Int,
    ctx: ExecContext = ExecContext(),
):
    """Parallelized vectorized X gate (swap amplitudes)."""
    var l = state.size()
    var stride = 1 << target
    var ptr_re = state.re_ptr()
    var ptr_im = state.im_ptr()

    var num_work_items = get_num_chunks(ctx)
    var total_blocks = l // (2 * stride)
    if total_blocks == 0:
        return
    var actual_workers = min(num_work_items, total_blocks)
    var blocks_per_worker = max(1, total_blocks // num_work_items)
    var threads = get_threads(ctx)

    @always_inline
    fn process_block(k: Int):
        @parameter
        fn vectorize_x[width: Int](m: Int):
            var idx = k + m
            var u_re = ptr_re.load[width=width](idx)
            var u_im = ptr_im.load[width=width](idx)
            var v_re = ptr_re.load[width=width](idx + stride)
            var v_im = ptr_im.load[width=width](idx + stride)
            ptr_re.store[width=width](idx, v_re)
            ptr_im.store[width=width](idx, v_im)
            ptr_re.store[width=width](idx + stride, u_re)
            ptr_im.store[width=width](idx + stride, u_im)

        if stride >= simd_width:
            vectorize[vectorize_x, simd_width](stride)
        else:
            for m in range(stride):
                var idx = k + m
                var u_re = ptr_re[idx]
                var u_im = ptr_im[idx]
                var v_re = ptr_re[idx + stride]
                var v_im = ptr_im[idx + stride]
                ptr_re[idx] = v_re
                ptr_im[idx] = v_im
                ptr_re[idx + stride] = u_re
                ptr_im[idx + stride] = u_im

    @parameter
    fn worker(item_id: Int):
        var start_block = item_id * blocks_per_worker
        var end_block = (
            total_blocks if item_id
            == actual_workers - 1 else (item_id + 1) * blocks_per_worker
        )
        for k_idx in range(start_block, end_block):
            process_block(k_idx * 2 * stride)

    if actual_workers > 1:
        if threads > 0:
            parallelize[worker](actual_workers, threads)
        else:
            parallelize[worker](actual_workers)
    else:
        worker(0)


@always_inline
fn transform_ry_simd_parallel(
    mut state: QuantumState,
    target: Int,
    theta: Float64,
    ctx: ExecContext = ExecContext(),
):
    """Parallelized vectorized RY gate."""
    var l = state.size()
    var stride = 1 << target
    var ptr_re = state.re_ptr()
    var ptr_im = state.im_ptr()
    var cos_t = cos(theta / 2)
    var sin_t = sin(theta / 2)

    var num_work_items = get_num_chunks(ctx)
    var total_blocks = l // (2 * stride)
    if total_blocks == 0:
        return
    var actual_workers = min(num_work_items, total_blocks)
    var blocks_per_worker = max(1, total_blocks // num_work_items)
    var threads = get_threads(ctx)

    @always_inline
    fn process_block(k: Int):
        @parameter
        fn vectorize_ry[width: Int](m: Int):
            var idx = k + m
            var u_re = ptr_re.load[width=width](idx)
            var u_im = ptr_im.load[width=width](idx)
            var v_re = ptr_re.load[width=width](idx + stride)
            var v_im = ptr_im.load[width=width](idx + stride)

            var out_u_re = u_re * cos_t - v_re * sin_t
            var out_u_im = u_im * cos_t - v_im * sin_t
            var out_v_re = u_re * sin_t + v_re * cos_t
            var out_v_im = u_im * sin_t + v_im * cos_t

            ptr_re.store[width=width](idx, out_u_re)
            ptr_im.store[width=width](idx, out_u_im)
            ptr_re.store[width=width](idx + stride, out_v_re)
            ptr_im.store[width=width](idx + stride, out_v_im)

        if stride >= simd_width:
            vectorize[vectorize_ry, simd_width](stride)
        else:
            for m in range(stride):
                var idx = k + m
                var u_re = ptr_re[idx]
                var u_im = ptr_im[idx]
                var v_re = ptr_re[idx + stride]
                var v_im = ptr_im[idx + stride]
                ptr_re[idx] = u_re * cos_t - v_re * sin_t
                ptr_im[idx] = u_im * cos_t - v_im * sin_t
                ptr_re[idx + stride] = u_re * sin_t + v_re * cos_t
                ptr_im[idx + stride] = u_im * sin_t + v_im * cos_t

    @parameter
    fn worker(item_id: Int):
        var start_block = item_id * blocks_per_worker
        var end_block = (
            total_blocks if item_id
            == actual_workers - 1 else (item_id + 1) * blocks_per_worker
        )
        for k_idx in range(start_block, end_block):
            process_block(k_idx * 2 * stride)

    if actual_workers > 1:
        if threads > 0:
            parallelize[worker](actual_workers, threads)
        else:
            parallelize[worker](actual_workers)
    else:
        worker(0)

@always_inline
fn transform_p_simd_parallel(
    mut state: QuantumState,
    target: Int,
    theta: Float64,
    ctx: ExecContext = ExecContext(),
):
    """Parallelized vectorized Phase gate (ref baseline)."""
    var l = state.size()
    var stride = 1 << target
    var ptr_re = state.re_ptr()
    var ptr_im = state.im_ptr()
    var cos_t = cos(theta)
    var sin_t = sin(theta)

    var num_work_items = get_num_chunks(ctx)
    var total_blocks = l // (2 * stride)
    if total_blocks == 0:
        return
    var actual_workers = min(num_work_items, total_blocks)
    var blocks_per_worker = max(1, total_blocks // num_work_items)
    var threads = get_threads(ctx)

    @always_inline
    fn process_block(k: Int):
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

    @parameter
    fn worker(item_id: Int):
        var start_block = item_id * blocks_per_worker
        var end_block = (
            total_blocks if item_id
            == actual_workers - 1 else (item_id + 1) * blocks_per_worker
        )
        for k_idx in range(start_block, end_block):
            process_block(k_idx * 2 * stride)

    if actual_workers > 1:
        if threads > 0:
            parallelize[worker](actual_workers, threads)
        else:
            parallelize[worker](actual_workers)
    else:
        worker(0)


fn transform_gate_simd_parallel(
    mut state: QuantumState,
    target: Int,
    gate: Gate,
    ctx: ExecContext = ExecContext(),
):
    """Parallelized generic SIMD gate with v2-style chunking."""
    var l = state.size()
    var stride = 1 << target
    var ptr_re = state.re_ptr()
    var ptr_im = state.im_ptr()

    var g00_re = gate[0][0].re
    var g00_im = gate[0][0].im
    var g01_re = gate[0][1].re
    var g01_im = gate[0][1].im
    var g10_re = gate[1][0].re
    var g10_im = gate[1][0].im
    var g11_re = gate[1][1].re
    var g11_im = gate[1][1].im

    var num_work_items = get_num_chunks(ctx)
    var total_blocks = l // (2 * stride)
    if total_blocks == 0:
        return
    var actual_workers = min(num_work_items, total_blocks)
    var blocks_per_worker = max(1, total_blocks // num_work_items)
    var threads = get_threads(ctx)

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
            fn vectorize_gate[width: Int](m: Int):
                var idx = k + m
                var u_re = ptr_re.load[width=width](idx)
                var u_im = ptr_im.load[width=width](idx)
                var v_re = ptr_re.load[width=width](idx + stride)
                var v_im = ptr_im.load[width=width](idx + stride)

                var out_u_re = (
                    u_re * g00_re
                    - u_im * g00_im
                    + v_re * g01_re
                    - v_im * g01_im
                )
                var out_u_im = (
                    u_re * g00_im
                    + u_im * g00_re
                    + v_re * g01_im
                    + v_im * g01_re
                )
                var out_v_re = (
                    u_re * g10_re
                    - u_im * g10_im
                    + v_re * g11_re
                    - v_im * g11_im
                )
                var out_v_im = (
                    u_re * g10_im
                    + u_im * g10_re
                    + v_re * g11_im
                    + v_im * g11_re
                )

                ptr_re.store[width=width](idx, out_u_re)
                ptr_im.store[width=width](idx, out_u_im)
                ptr_re.store[width=width](idx + stride, out_v_re)
                ptr_im.store[width=width](idx + stride, out_v_im)

            if stride >= simd_width:
                vectorize[vectorize_gate, simd_width](stride)
            else:
                for m in range(stride):
                    var idx = k + m
                    var u_re = ptr_re[idx]
                    var u_im = ptr_im[idx]
                    var v_re = ptr_re[idx + stride]
                    var v_im = ptr_im[idx + stride]

                    ptr_re[idx] = (
                        u_re * g00_re
                        - u_im * g00_im
                        + v_re * g01_re
                        - v_im * g01_im
                    )
                    ptr_im[idx] = (
                        u_re * g00_im
                        + u_im * g00_re
                        + v_re * g01_im
                        + v_im * g01_re
                    )
                    ptr_re[idx + stride] = (
                        u_re * g10_re
                        - u_im * g10_im
                        + v_re * g11_re
                        - v_im * g11_im
                    )
                    ptr_im[idx + stride] = (
                        u_re * g10_im
                        + u_im * g10_re
                        + v_re * g11_im
                        + v_im * g11_re
                    )

    if actual_workers > 1:
        if threads > 0:
            parallelize[worker](actual_workers, threads)
        else:
            parallelize[worker](actual_workers)
    else:
        worker(0)


fn mc_transform_simd_parallel(
    mut state: QuantumState,
    controls: List[Int],
    target: Int,
    gate: Gate,
    ctx: ExecContext = ExecContext(),
):
    """Parallel multi-controlled gate (mask check per index)."""
    if len(controls) == 0:
        transform_gate_simd_parallel(state, target, gate, ctx)
        return

    var mask = 0
    for c in controls:
        mask |= 1 << c

    var l = state.size()
    var stride = 1 << target
    var ptr_re = state.re_ptr()
    var ptr_im = state.im_ptr()

    var g00_re = gate[0][0].re
    var g00_im = gate[0][0].im
    var g01_re = gate[0][1].re
    var g01_im = gate[0][1].im
    var g10_re = gate[1][0].re
    var g10_im = gate[1][0].im
    var g11_re = gate[1][1].re
    var g11_im = gate[1][1].im

    var num_work_items = get_num_chunks(ctx)
    var total_blocks = l // (2 * stride)
    if total_blocks == 0:
        return
    var actual_workers = min(num_work_items, total_blocks)
    var blocks_per_worker = max(1, total_blocks // num_work_items)
    var threads = get_threads(ctx)

    @parameter
    fn worker(item_id: Int):
        var start_block = item_id * blocks_per_worker
        var end_block = (
            total_blocks if item_id
            == actual_workers - 1 else (item_id + 1) * blocks_per_worker
        )
        for k_idx in range(start_block, end_block):
            var k = k_idx * 2 * stride
            for m in range(stride):
                var idx = k + m
                if (idx & mask) != mask:
                    continue
                var u_re = ptr_re[idx]
                var u_im = ptr_im[idx]
                var v_re = ptr_re[idx + stride]
                var v_im = ptr_im[idx + stride]

                ptr_re[idx] = (
                    u_re * g00_re
                    - u_im * g00_im
                    + v_re * g01_re
                    - v_im * g01_im
                )
                ptr_im[idx] = (
                    u_re * g00_im
                    + u_im * g00_re
                    + v_re * g01_im
                    + v_im * g01_re
                )
                ptr_re[idx + stride] = (
                    u_re * g10_re
                    - u_im * g10_im
                    + v_re * g11_re
                    - v_im * g11_im
                )
                ptr_im[idx + stride] = (
                    u_re * g10_im
                    + u_im * g10_re
                    + v_re * g11_im
                    + v_im * g11_re
                )

    if actual_workers > 1:
        if threads > 0:
            parallelize[worker](actual_workers, threads)
        else:
            parallelize[worker](actual_workers)
    else:
        worker(0)


@always_inline
fn c_transform_x_simd_parallel(
    mut state: QuantumState,
    control: Int,
    target: Int,
    ctx: ExecContext = ExecContext(),
):
    """Parallelized controlled X gate using vectorized mask."""
    var l = state.size()
    var stride = 1 << target
    var ptr_re = state.re_ptr()
    var ptr_im = state.im_ptr()

    var num_work_items = get_num_chunks(ctx)
    var total_blocks = l // (2 * stride)
    if total_blocks == 0:
        return
    var actual_workers = min(num_work_items, total_blocks)
    var blocks_per_worker = max(1, total_blocks // num_work_items)
    var threads = get_threads(ctx)

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
                var idx = k + m
                var offsets = SIMD[DType.int64, width]()
                for i in range(width):
                    offsets[i] = i
                var indices = SIMD[DType.int64, width](idx) + offsets
                var mask_val = indices & (1 << control)
                var mask = mask_val.cast[DType.bool]()

                var u_re = ptr_re.load[width=width](idx)
                var u_im = ptr_im.load[width=width](idx)
                var v_re = ptr_re.load[width=width](idx + stride)
                var v_im = ptr_im.load[width=width](idx + stride)

                var out_u_re = mask.select(v_re, u_re)
                var out_u_im = mask.select(v_im, u_im)
                var out_v_re = mask.select(u_re, v_re)
                var out_v_im = mask.select(u_im, v_im)

                ptr_re.store[width=width](idx, out_u_re)
                ptr_im.store[width=width](idx, out_u_im)
                ptr_re.store[width=width](idx + stride, out_v_re)
                ptr_im.store[width=width](idx + stride, out_v_im)

            if stride >= simd_width:
                vectorize[vectorize_x, simd_width](stride)
            else:
                for m in range(stride):
                    var idx = k + m
                    if (idx & (1 << control)) != 0:
                        var u_re = ptr_re[idx]
                        var u_im = ptr_im[idx]
                        var v_re = ptr_re[idx + stride]
                        var v_im = ptr_im[idx + stride]
                        ptr_re[idx] = v_re
                        ptr_im[idx] = v_im
                        ptr_re[idx + stride] = u_re
                        ptr_im[idx + stride] = u_im

    if actual_workers > 1:
        if threads > 0:
            parallelize[worker](actual_workers, threads)
        else:
            parallelize[worker](actual_workers)
    else:
        worker(0)


@always_inline
fn c_transform_ry_simd_parallel(
    mut state: QuantumState,
    control: Int,
    target: Int,
    theta: Float64,
    ctx: ExecContext = ExecContext(),
):
    """Parallelized controlled RY gate using vectorized mask."""
    var l = state.size()
    var stride = 1 << target
    var ptr_re = state.re_ptr()
    var ptr_im = state.im_ptr()
    var cos_t = cos(theta / 2)
    var sin_t = sin(theta / 2)

    var num_work_items = get_num_chunks(ctx)
    var total_blocks = l // (2 * stride)
    if total_blocks == 0:
        return
    var actual_workers = min(num_work_items, total_blocks)
    var blocks_per_worker = max(1, total_blocks // num_work_items)
    var threads = get_threads(ctx)

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
            fn vectorize_ry[width: Int](m: Int):
                var idx = k + m
                var offsets = SIMD[DType.int64, width]()
                for i in range(width):
                    offsets[i] = i
                var indices = SIMD[DType.int64, width](idx) + offsets
                var mask_val = indices & (1 << control)
                var mask = mask_val.cast[DType.bool]()

                var u_re = ptr_re.load[width=width](idx)
                var u_im = ptr_im.load[width=width](idx)
                var v_re = ptr_re.load[width=width](idx + stride)
                var v_im = ptr_im.load[width=width](idx + stride)

                var out_u_re = u_re * cos_t - v_re * sin_t
                var out_u_im = u_im * cos_t - v_im * sin_t
                var out_v_re = u_re * sin_t + v_re * cos_t
                var out_v_im = u_im * sin_t + v_im * cos_t

                out_u_re = mask.select(out_u_re, u_re)
                out_u_im = mask.select(out_u_im, u_im)
                out_v_re = mask.select(out_v_re, v_re)
                out_v_im = mask.select(out_v_im, v_im)

                ptr_re.store[width=width](idx, out_u_re)
                ptr_im.store[width=width](idx, out_u_im)
                ptr_re.store[width=width](idx + stride, out_v_re)
                ptr_im.store[width=width](idx + stride, out_v_im)

            if stride >= simd_width:
                vectorize[vectorize_ry, simd_width](stride)
            else:
                for m in range(stride):
                    var idx = k + m
                    if (idx & (1 << control)) != 0:
                        var u_re = ptr_re[idx]
                        var u_im = ptr_im[idx]
                        var v_re = ptr_re[idx + stride]
                        var v_im = ptr_im[idx + stride]
                        ptr_re[idx] = u_re * cos_t - v_re * sin_t
                        ptr_im[idx] = u_im * cos_t - v_im * sin_t
                        ptr_re[idx + stride] = u_re * sin_t + v_re * cos_t
                        ptr_im[idx + stride] = u_im * sin_t + v_im * cos_t

    if actual_workers > 1:
        if threads > 0:
            parallelize[worker](actual_workers, threads)
        else:
            parallelize[worker](actual_workers)
    else:
        worker(0)



@always_inline
fn c_transform_p_simd_parallel(
    mut state: QuantumState,
    control: Int,
    target: Int,
    theta: Float64,
    ctx: ExecContext = ExecContext(),
):
    """Specialized CP gate with v2-style chunking and parallelization."""
    var c_stride = 1 << control
    var t_stride = 1 << target
    var size = state.size()
    var ptr_re = state.re_ptr()
    var ptr_im = state.im_ptr()

    var cos_t = cos(theta)
    var sin_t = sin(theta)

    var num_work_items = get_num_chunks(ctx)
    if num_work_items == 0:
        num_work_items = 16
    var threads = get_threads(ctx)

    if target < control:
        var total_work = (size // (2 * c_stride)) * (c_stride // (2 * t_stride))
        var actual_work_items = min(num_work_items, total_work)
        var chunk_size = max(1, total_work // num_work_items)

        @parameter
        fn worker_low(item_id: Int):
            var start_idx = item_id * chunk_size
            var end_idx = (
                total_work if item_id
                == actual_work_items - 1 else (item_id + 1) * chunk_size
            )

            var segments_per_block = c_stride // (2 * t_stride)
            for s in range(start_idx, end_idx):
                var k = s // segments_per_block
                var j = s % segments_per_block
                var sub_start = k * 2 * c_stride + c_stride + j * 2 * t_stride

                @parameter
                fn vectorize_p[width: Int](m: Int):
                    var idx = sub_start + m + t_stride
                    var v_re = ptr_re.load[width=width](idx)
                    var v_im = ptr_im.load[width=width](idx)
                    var res_re = v_re * cos_t - v_im * sin_t
                    var res_im = v_re * sin_t + v_im * cos_t
                    ptr_re.store[width=width](idx, res_re)
                    ptr_im.store[width=width](idx, res_im)

                if t_stride >= simd_width:
                    vectorize[vectorize_p, simd_width](t_stride)
                else:
                    for m in range(t_stride):
                        var idx = sub_start + m + t_stride
                        var v_re = ptr_re[idx]
                        var v_im = ptr_im[idx]
                        ptr_re[idx] = v_re * cos_t - v_im * sin_t
                        ptr_im[idx] = v_re * sin_t + v_im * cos_t

        if actual_work_items > 1:
            if threads > 0:
                parallelize[worker_low](actual_work_items, threads)
            else:
                parallelize[worker_low](actual_work_items)
        else:
            worker_low(0)
    else:
        var total_work = (size // (2 * t_stride)) * (t_stride // (2 * c_stride))
        var actual_work_items = min(num_work_items, total_work)
        var chunk_size = max(1, total_work // num_work_items)

        @parameter
        fn worker_high(item_id: Int):
            var start_idx = item_id * chunk_size
            var end_idx = (
                total_work if item_id
                == actual_work_items - 1 else (item_id + 1) * chunk_size
            )

            var segments_per_block = t_stride // (2 * c_stride)
            for s in range(start_idx, end_idx):
                var k = s // segments_per_block
                var p = s % segments_per_block
                var p_start = k * 2 * t_stride + p * 2 * c_stride + c_stride

                @parameter
                fn vectorize_p[width: Int](m: Int):
                    var idx = p_start + m + t_stride
                    var v_re = ptr_re.load[width=width](idx)
                    var v_im = ptr_im.load[width=width](idx)
                    var res_re = v_re * cos_t - v_im * sin_t
                    var res_im = v_re * sin_t + v_im * cos_t
                    ptr_re.store[width=width](idx, res_re)
                    ptr_im.store[width=width](idx, res_im)

                if c_stride >= simd_width:
                    vectorize[vectorize_p, simd_width](c_stride)
                else:
                    for m in range(c_stride):
                        var idx = p_start + m + t_stride
                        var v_re = ptr_re[idx]
                        var v_im = ptr_im[idx]
                        ptr_re[idx] = v_re * cos_t - v_im * sin_t
                        ptr_im[idx] = v_re * sin_t + v_im * cos_t

        if actual_work_items > 1:
            if threads > 0:
                parallelize[worker_high](actual_work_items, threads)
            else:
                parallelize[worker_high](actual_work_items)
        else:
            worker_high(0)
