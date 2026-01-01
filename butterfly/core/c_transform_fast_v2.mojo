"""
Specialized controlled gate kernels v2.
Optimized version of common controlled operations with chunked parallelization.
"""

from butterfly.core.state import QuantumState, simd_width
from butterfly.utils.config import get_workers
from algorithm import vectorize, parallelize
from math import cos, sin

alias sq_half = 0.7071067811865476  # 1/sqrt(2)


@always_inline
fn c_transform_h_simd_v2(mut state: QuantumState, control: Int, target: Int):
    """Specialized CH gate v2 with chunked parallelization."""
    var c_stride = 1 << control
    var t_stride = 1 << target
    var size = state.size()
    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()

    # Use chunked parallelization (configurable via butterfly.config)
    var num_work_items = get_workers("quantum_simd_v2_chunks")
    if num_work_items == 0:
        num_work_items = 16  # Default if not configured

    if target < control:
        var num_c_blocks = size // (2 * c_stride)
        var total_work = num_c_blocks * (c_stride // (2 * t_stride))
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
                fn vectorize_h[width: Int](m: Int):
                    var idx = sub_start + m
                    var u_re = ptr_re.load[width=width](idx)
                    var u_im = ptr_im.load[width=width](idx)
                    var v_re = ptr_re.load[width=width](idx + t_stride)
                    var v_im = ptr_im.load[width=width](idx + t_stride)

                    var sum_re = (u_re + v_re) * sq_half
                    var sum_im = (u_im + v_im) * sq_half
                    var diff_re = (u_re - v_re) * sq_half
                    var diff_im = (u_im - v_im) * sq_half

                    ptr_re.store[width=width](idx, sum_re)
                    ptr_im.store[width=width](idx, sum_im)
                    ptr_re.store[width=width](idx + t_stride, diff_re)
                    ptr_im.store[width=width](idx + t_stride, diff_im)

                if t_stride >= simd_width:
                    vectorize[vectorize_h, simd_width](t_stride)
                else:
                    for m in range(t_stride):
                        var idx = sub_start + m
                        var u_re = ptr_re[idx]
                        var u_im = ptr_im[idx]
                        var v_re = ptr_re[idx + t_stride]
                        var v_im = ptr_im[idx + t_stride]
                        ptr_re[idx] = (u_re + v_re) * sq_half
                        ptr_im[idx] = (u_im + v_im) * sq_half
                        ptr_re[idx + t_stride] = (u_re - v_re) * sq_half
                        ptr_im[idx + t_stride] = (u_im - v_im) * sq_half

        parallelize[worker_low](actual_work_items)
    else:
        var num_t_blocks = size // (2 * t_stride)
        var total_work = num_t_blocks * (t_stride // (2 * c_stride))
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
                fn vectorize_h[width: Int](m: Int):
                    var idx = p_start + m
                    var u_re = ptr_re.load[width=width](idx)
                    var u_im = ptr_im.load[width=width](idx)
                    var v_re = ptr_re.load[width=width](idx + t_stride)
                    var v_im = ptr_im.load[width=width](idx + t_stride)

                    var sum_re = (u_re + v_re) * sq_half
                    var sum_im = (u_im + v_im) * sq_half
                    var diff_re = (u_re - v_re) * sq_half
                    var diff_im = (u_im - v_im) * sq_half

                    ptr_re.store[width=width](idx, sum_re)
                    ptr_im.store[width=width](idx, sum_im)
                    ptr_re.store[width=width](idx + t_stride, diff_re)
                    ptr_im.store[width=width](idx + t_stride, diff_im)

                if c_stride >= simd_width:
                    vectorize[vectorize_h, simd_width](c_stride)
                else:
                    for m in range(c_stride):
                        var idx = p_start + m
                        var u_re = ptr_re[idx]
                        var u_im = ptr_im[idx]
                        var v_re = ptr_re[idx + t_stride]
                        var v_im = ptr_im[idx + t_stride]
                        ptr_re[idx] = (u_re + v_re) * sq_half
                        ptr_im[idx] = (u_im + v_im) * sq_half
                        ptr_re[idx + t_stride] = (u_re - v_re) * sq_half
                        ptr_im[idx + t_stride] = (u_im - v_im) * sq_half

        parallelize[worker_high](actual_work_items)


@always_inline
fn c_transform_x_simd_v2(mut state: QuantumState, control: Int, target: Int):
    """Specialized CX gate v2 with chunked parallelization."""
    var c_stride = 1 << control
    var t_stride = 1 << target
    var size = state.size()
    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()

    var num_work_items = get_workers("quantum_simd_v2_chunks")
    if num_work_items == 0:
        num_work_items = 16  # Default if not configured

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
                fn vectorize_swap[width: Int](m: Int):
                    var idx = sub_start + m
                    var u_re = ptr_re.load[width=width](idx)
                    var u_im = ptr_im.load[width=width](idx)
                    var v_re = ptr_re.load[width=width](idx + t_stride)
                    var v_im = ptr_im.load[width=width](idx + t_stride)
                    ptr_re.store[width=width](idx, v_re)
                    ptr_im.store[width=width](idx, v_im)
                    ptr_re.store[width=width](idx + t_stride, u_re)
                    ptr_im.store[width=width](idx + t_stride, u_im)

                if t_stride >= simd_width:
                    vectorize[vectorize_swap, simd_width](t_stride)
                else:
                    for m in range(t_stride):
                        var idx = sub_start + m
                        var tmp_re = ptr_re[idx]
                        var tmp_im = ptr_im[idx]
                        ptr_re[idx] = ptr_re[idx + t_stride]
                        ptr_im[idx] = ptr_im[idx + t_stride]
                        ptr_re[idx + t_stride] = tmp_re
                        ptr_im[idx + t_stride] = tmp_im

        parallelize[worker_low](actual_work_items)
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
                fn vectorize_swap[width: Int](m: Int):
                    var idx = p_start + m
                    var u_re = ptr_re.load[width=width](idx)
                    var u_im = ptr_im.load[width=width](idx)
                    var v_re = ptr_re.load[width=width](idx + t_stride)
                    var v_im = ptr_im.load[width=width](idx + t_stride)
                    ptr_re.store[width=width](idx, v_re)
                    ptr_im.store[width=width](idx, v_im)
                    ptr_re.store[width=width](idx + t_stride, u_re)
                    ptr_im.store[width=width](idx + t_stride, u_im)

                if c_stride >= simd_width:
                    vectorize[vectorize_swap, simd_width](c_stride)
                else:
                    for m in range(c_stride):
                        var idx = p_start + m
                        var tmp_re = ptr_re[idx]
                        var tmp_im = ptr_im[idx]
                        ptr_re[idx] = ptr_re[idx + t_stride]
                        ptr_im[idx] = ptr_im[idx + t_stride]
                        ptr_re[idx + t_stride] = tmp_re
                        ptr_im[idx + t_stride] = tmp_im

        parallelize[worker_high](actual_work_items)


@always_inline
fn c_transform_p_simd_v2(
    mut state: QuantumState, control: Int, target: Int, theta: Float64
):
    """Specialized CP gate v2 with chunked parallelization."""
    var c_stride = 1 << control
    var t_stride = 1 << target
    var size = state.size()
    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()

    var cos_t = cos(theta)
    var sin_t = sin(theta)

    var num_work_items = get_workers("quantum_simd_v2_chunks")
    if num_work_items == 0:
        num_work_items = 16  # Default if not configured

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
                    var idx = (
                        sub_start + m + t_stride
                    )  # Only target=1 is affected
                    var v_re = ptr_re.load[width=width](idx)
                    var v_im = ptr_im.load[width=width](idx)

                    # v' = (v_re + i*v_im)*(cos + i*sin) = (v_re*cos - v_im*sin) + i(v_re*sin + v_im*cos)
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

        parallelize[worker_low](actual_work_items)
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
                    var idx = p_start + m + t_stride  # Only target=1 affected
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

        parallelize[worker_high](actual_work_items)


@always_inline
fn c_transform_y_simd_v2(mut state: QuantumState, control: Int, target: Int):
    """Specialized CY gate v2 with chunked parallelization."""
    var c_stride = 1 << control
    var t_stride = 1 << target
    var size = state.size()
    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()

    var num_work_items = get_workers("quantum_simd_v2_chunks")
    if num_work_items == 0:
        num_work_items = 16

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
                fn vectorize_y[width: Int](m: Int):
                    var idx0 = sub_start + m
                    var idx1 = idx0 + t_stride
                    var u_re = ptr_re.load[width=width](idx0)
                    var u_im = ptr_im.load[width=width](idx0)
                    var v_re = ptr_re.load[width=width](idx1)
                    var v_im = ptr_im.load[width=width](idx1)
                    # Y: [u, v] -> [v_im - i*v_re, -u_im + i*u_re]
                    ptr_re.store[width=width](idx0, v_im)
                    ptr_im.store[width=width](idx0, -v_re)
                    ptr_re.store[width=width](idx1, -u_im)
                    ptr_im.store[width=width](idx1, u_re)

                if t_stride >= simd_width:
                    vectorize[vectorize_y, simd_width](t_stride)
                else:
                    for m in range(t_stride):
                        var idx0 = sub_start + m
                        var idx1 = idx0 + t_stride
                        var u_re = ptr_re[idx0]
                        var u_im = ptr_im[idx0]
                        var v_re = ptr_re[idx1]
                        var v_im = ptr_im[idx1]
                        ptr_re[idx0] = v_im
                        ptr_im[idx0] = -v_re
                        ptr_re[idx1] = -u_im
                        ptr_im[idx1] = u_re

        parallelize[worker_low](actual_work_items)
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
                fn vectorize_y[width: Int](m: Int):
                    var idx0 = p_start + m
                    var idx1 = idx0 + t_stride
                    var u_re = ptr_re.load[width=width](idx0)
                    var u_im = ptr_im.load[width=width](idx0)
                    var v_re = ptr_re.load[width=width](idx1)
                    var v_im = ptr_im.load[width=width](idx1)
                    ptr_re.store[width=width](idx0, v_im)
                    ptr_im.store[width=width](idx0, -v_re)
                    ptr_re.store[width=width](idx1, -u_im)
                    ptr_im.store[width=width](idx1, u_re)

                if c_stride >= simd_width:
                    vectorize[vectorize_y, simd_width](c_stride)
                else:
                    for m in range(c_stride):
                        var idx0 = p_start + m
                        var idx1 = idx0 + t_stride
                        var u_re = ptr_re[idx0]
                        var u_im = ptr_im[idx0]
                        var v_re = ptr_re[idx1]
                        var v_im = ptr_im[idx1]
                        ptr_re[idx0] = v_im
                        ptr_im[idx0] = -v_re
                        ptr_re[idx1] = -u_im
                        ptr_im[idx1] = u_re

        parallelize[worker_high](actual_work_items)


@always_inline
fn c_transform_rz_simd_v2(
    mut state: QuantumState, control: Int, target: Int, theta: Float64
):
    """Specialized CRZ gate v2 with chunked parallelization."""
    var c_stride = 1 << control
    var t_stride = 1 << target
    var size = state.size()
    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()

    var phi = theta / 2.0
    var cos_p = cos(phi)
    var sin_p = sin(phi)

    var num_work_items = get_workers("quantum_simd_v2_chunks")
    if num_work_items == 0:
        num_work_items = 16

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
                fn vectorize_rz[width: Int](m: Int):
                    var idx0 = sub_start + m
                    var idx1 = idx0 + t_stride
                    var u_re = ptr_re.load[width=width](idx0)
                    var u_im = ptr_im.load[width=width](idx0)
                    var v_re = ptr_re.load[width=width](idx1)
                    var v_im = ptr_im.load[width=width](idx1)
                    ptr_re.store[width=width](idx0, u_re * cos_p + u_im * sin_p)
                    ptr_im.store[width=width](idx0, u_im * cos_p - u_re * sin_p)
                    ptr_re.store[width=width](idx1, v_re * cos_p - v_im * sin_p)
                    ptr_im.store[width=width](idx1, v_im * cos_p + v_re * sin_p)

                if t_stride >= simd_width:
                    vectorize[vectorize_rz, simd_width](t_stride)
                else:
                    for m in range(t_stride):
                        var idx0 = sub_start + m
                        var idx1 = idx0 + t_stride
                        var u_re = ptr_re[idx0]
                        var u_im = ptr_im[idx0]
                        var v_re = ptr_re[idx1]
                        var v_im = ptr_im[idx1]
                        ptr_re[idx0] = u_re * cos_p + u_im * sin_p
                        ptr_im[idx0] = u_im * cos_p - u_re * sin_p
                        ptr_re[idx1] = v_re * cos_p - v_im * sin_p
                        ptr_im[idx1] = v_im * cos_p + v_re * sin_p

        parallelize[worker_low](actual_work_items)
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
                fn vectorize_rz[width: Int](m: Int):
                    var idx0 = p_start + m
                    var idx1 = idx0 + t_stride
                    var u_re = ptr_re.load[width=width](idx0)
                    var u_im = ptr_im.load[width=width](idx0)
                    var v_re = ptr_re.load[width=width](idx1)
                    var v_im = ptr_im.load[width=width](idx1)
                    ptr_re.store[width=width](idx0, u_re * cos_p + u_im * sin_p)
                    ptr_im.store[width=width](idx0, u_im * cos_p - u_re * sin_p)
                    ptr_re.store[width=width](idx1, v_re * cos_p - v_im * sin_p)
                    ptr_im.store[width=width](idx1, v_im * cos_p + v_re * sin_p)

                if c_stride >= simd_width:
                    vectorize[vectorize_rz, simd_width](c_stride)
                else:
                    for m in range(c_stride):
                        var idx0 = p_start + m
                        var idx1 = idx0 + t_stride
                        var u_re = ptr_re[idx0]
                        var u_im = ptr_im[idx0]
                        var v_re = ptr_re[idx1]
                        var v_im = ptr_im[idx1]
                        ptr_re[idx0] = u_re * cos_p + u_im * sin_p
                        ptr_im[idx0] = u_im * cos_p - u_re * sin_p
                        ptr_re[idx1] = v_re * cos_p - v_im * sin_p
                        ptr_im[idx1] = v_im * cos_p + v_re * sin_p

        parallelize[worker_high](actual_work_items)


@always_inline
fn c_transform_rx_simd_v2(
    mut state: QuantumState, control: Int, target: Int, theta: Float64
):
    """Specialized CRX gate v2 with chunked parallelization."""
    var c_stride = 1 << control
    var t_stride = 1 << target
    var size = state.size()
    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()

    var phi = theta / 2.0
    var cos_p = cos(phi)
    var sin_p = sin(phi)

    var num_work_items = get_workers("quantum_simd_v2_chunks")
    if num_work_items == 0:
        num_work_items = 16

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
                fn vectorize_rx[width: Int](m: Int):
                    var idx0 = sub_start + m
                    var idx1 = idx0 + t_stride
                    var u_re = ptr_re.load[width=width](idx0)
                    var u_im = ptr_im.load[width=width](idx0)
                    var v_re = ptr_re.load[width=width](idx1)
                    var v_im = ptr_im.load[width=width](idx1)
                    # u -> cos*u - i*sin*v, v -> -i*sin*u + cos*v
                    ptr_re.store[width=width](idx0, u_re * cos_p + v_im * sin_p)
                    ptr_im.store[width=width](idx0, u_im * cos_p - v_re * sin_p)
                    ptr_re.store[width=width](idx1, v_re * cos_p + u_im * sin_p)
                    ptr_im.store[width=width](idx1, v_im * cos_p - u_re * sin_p)

                if t_stride >= simd_width:
                    vectorize[vectorize_rx, simd_width](t_stride)
                else:
                    for m in range(t_stride):
                        var idx0 = sub_start + m
                        var idx1 = idx0 + t_stride
                        var u_re = ptr_re[idx0]
                        var u_im = ptr_im[idx0]
                        var v_re = ptr_re[idx1]
                        var v_im = ptr_im[idx1]
                        ptr_re[idx0] = u_re * cos_p + v_im * sin_p
                        ptr_im[idx0] = u_im * cos_p - v_re * sin_p
                        ptr_re[idx1] = v_re * cos_p + u_im * sin_p
                        ptr_im[idx1] = v_im * cos_p - u_re * sin_p

        parallelize[worker_low](actual_work_items)
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
                fn vectorize_rx[width: Int](m: Int):
                    var idx0 = p_start + m
                    var idx1 = idx0 + t_stride
                    var u_re = ptr_re.load[width=width](idx0)
                    var u_im = ptr_im.load[width=width](idx0)
                    var v_re = ptr_re.load[width=width](idx1)
                    var v_im = ptr_im.load[width=width](idx1)
                    ptr_re.store[width=width](idx0, u_re * cos_p + v_im * sin_p)
                    ptr_im.store[width=width](idx0, u_im * cos_p - v_re * sin_p)
                    ptr_re.store[width=width](idx1, v_re * cos_p + u_im * sin_p)
                    ptr_im.store[width=width](idx1, v_im * cos_p - u_re * sin_p)

                if c_stride >= simd_width:
                    vectorize[vectorize_rx, simd_width](c_stride)
                else:
                    for m in range(c_stride):
                        var idx0 = p_start + m
                        var idx1 = idx0 + t_stride
                        var u_re = ptr_re[idx0]
                        var u_im = ptr_im[idx0]
                        var v_re = ptr_re[idx1]
                        var v_im = ptr_im[idx1]
                        ptr_re[idx0] = u_re * cos_p + v_im * sin_p
                        ptr_im[idx0] = u_im * cos_p - v_re * sin_p
                        ptr_re[idx1] = v_re * cos_p + u_im * sin_p
                        ptr_im[idx1] = v_im * cos_p - u_re * sin_p

        parallelize[worker_high](actual_work_items)


@always_inline
fn c_transform_ry_simd_v2(
    mut state: QuantumState, control: Int, target: Int, theta: Float64
):
    """Specialized CRY gate v2 with chunked parallelization."""
    var c_stride = 1 << control
    var t_stride = 1 << target
    var size = state.size()
    var ptr_re = state.re.unsafe_ptr()
    var ptr_im = state.im.unsafe_ptr()

    var phi = theta / 2.0
    var cos_p = cos(phi)
    var sin_p = sin(phi)

    var num_work_items = get_workers("quantum_simd_v2_chunks")
    if num_work_items == 0:
        num_work_items = 16

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
                fn vectorize_ry[width: Int](m: Int):
                    var idx0 = sub_start + m
                    var idx1 = idx0 + t_stride
                    var u_re = ptr_re.load[width=width](idx0)
                    var u_im = ptr_im.load[width=width](idx0)
                    var v_re = ptr_re.load[width=width](idx1)
                    var v_im = ptr_im.load[width=width](idx1)
                    # u -> cos*u - sin*v, v -> sin*u + cos*v
                    ptr_re.store[width=width](idx0, u_re * cos_p - v_re * sin_p)
                    ptr_im.store[width=width](idx0, u_im * cos_p - v_im * sin_p)
                    ptr_re.store[width=width](idx1, v_re * cos_p + u_re * sin_p)
                    ptr_im.store[width=width](idx1, v_im * cos_p + u_im * sin_p)

                if t_stride >= simd_width:
                    vectorize[vectorize_ry, simd_width](t_stride)
                else:
                    for m in range(t_stride):
                        var idx0 = sub_start + m
                        var idx1 = idx0 + t_stride
                        var u_re = ptr_re[idx0]
                        var u_im = ptr_im[idx0]
                        var v_re = ptr_re[idx1]
                        var v_im = ptr_im[idx1]
                        ptr_re[idx0] = u_re * cos_p - v_re * sin_p
                        ptr_im[idx0] = u_im * cos_p - v_im * sin_p
                        ptr_re[idx1] = v_re * cos_p + u_re * sin_p
                        ptr_im[idx1] = v_im * cos_p + u_im * sin_p

        parallelize[worker_low](actual_work_items)
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
                fn vectorize_ry[width: Int](m: Int):
                    var idx0 = p_start + m
                    var idx1 = idx0 + t_stride
                    var u_re = ptr_re.load[width=width](idx0)
                    var u_im = ptr_im.load[width=width](idx0)
                    var v_re = ptr_re.load[width=width](idx1)
                    var v_im = ptr_im.load[width=width](idx1)
                    ptr_re.store[width=width](idx0, u_re * cos_p - v_re * sin_p)
                    ptr_im.store[width=width](idx0, u_im * cos_p - v_im * sin_p)
                    ptr_re.store[width=width](idx1, v_re * cos_p + u_re * sin_p)
                    ptr_im.store[width=width](idx1, v_im * cos_p + u_im * sin_p)

                if c_stride >= simd_width:
                    vectorize[vectorize_ry, simd_width](c_stride)
                else:
                    for m in range(c_stride):
                        var idx0 = p_start + m
                        var idx1 = idx0 + t_stride
                        var u_re = ptr_re[idx0]
                        var u_im = ptr_im[idx0]
                        var v_re = ptr_re[idx1]
                        var v_im = ptr_im[idx1]
                        ptr_re[idx0] = u_re * cos_p - v_re * sin_p
                        ptr_im[idx0] = u_im * cos_p - v_im * sin_p
                        ptr_re[idx1] = v_re * cos_p + u_re * sin_p
                        ptr_im[idx1] = v_im * cos_p + u_im * sin_p

        parallelize[worker_high](actual_work_items)
