from butterfly.core.state import QuantumState
from butterfly.core.types import *
from butterfly.core.gates import *
from butterfly.utils.context import ExecContext
from math import cos, sin
from algorithm import vectorize


fn transform_simd(
    mut state: QuantumState,
    target: Int,
    gate: Gate,
    gate_kind: Int = -1,
    gate_arg: FloatType = 0,
    ctx: ExecContext = ExecContext(),
):
    """Generic SIMD single-qubit transform using block iteration."""
    if gate_kind == GateKind.H and ctx.simd_use_specialized_h:
        transform_h_simd(state, target)
        return
    if gate_kind == GateKind.P and ctx.simd_use_specialized_p:
        transform_p_simd(state, target, Float64(gate_arg))
        return
    if gate_kind == GateKind.X and ctx.simd_use_specialized_x:
        transform_x_simd(state, target)
        return
    if gate_kind == GateKind.RY and ctx.simd_use_specialized_ry:
        transform_ry_simd(state, target, Float64(gate_arg))
        return

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

    for k in range(0, l, 2 * stride):

        @parameter
        fn vectorize_gate[width: Int](m: Int):
            var idx = k + m
            var u_re = ptr_re.load[width=width](idx)
            var u_im = ptr_im.load[width=width](idx)
            var v_re = ptr_re.load[width=width](idx + stride)
            var v_im = ptr_im.load[width=width](idx + stride)

            var out_u_re = u_re * g00_re - u_im * g00_im + v_re * g01_re - v_im * g01_im
            var out_u_im = u_re * g00_im + u_im * g00_re + v_re * g01_im + v_im * g01_re
            var out_v_re = u_re * g10_re - u_im * g10_im + v_re * g11_re - v_im * g11_im
            var out_v_im = u_re * g10_im + u_im * g10_re + v_re * g11_im + v_im * g11_re

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

                ptr_re[idx] = u_re * g00_re - u_im * g00_im + v_re * g01_re - v_im * g01_im
                ptr_im[idx] = u_re * g00_im + u_im * g00_re + v_re * g01_im + v_im * g01_re
                ptr_re[idx + stride] = u_re * g10_re - u_im * g10_im + v_re * g11_re - v_im * g11_im
                ptr_im[idx + stride] = u_re * g10_im + u_im * g10_re + v_re * g11_im + v_im * g11_re


fn transform_h_simd(mut state: QuantumState, target: Int):
    """Single-threaded SIMD Hadamard transform."""
    var l = state.size()
    var stride = 1 << target
    var ptr_re = state.re_ptr()
    var ptr_im = state.im_ptr()

    for k in range(0, l, 2 * stride):

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


fn transform_x_simd(mut state: QuantumState, target: Int):
    """Single-threaded SIMD X transform (swap amplitudes)."""
    var l = state.size()
    var stride = 1 << target
    var ptr_re = state.re_ptr()
    var ptr_im = state.im_ptr()

    for k in range(0, l, 2 * stride):

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


fn transform_ry_simd(mut state: QuantumState, target: Int, theta: Float64):
    """Single-threaded SIMD RY transform."""
    var l = state.size()
    var stride = 1 << target
    var ptr_re = state.re_ptr()
    var ptr_im = state.im_ptr()
    var cos_t = cos(theta / 2)
    var sin_t = sin(theta / 2)

    for k in range(0, l, 2 * stride):

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


fn transform_p_simd(mut state: QuantumState, target: Int, theta: Float64):
    """Single-threaded SIMD Phase transform."""
    var l = state.size()
    var stride = 1 << target
    var ptr_re = state.re_ptr()
    var ptr_im = state.im_ptr()
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
                ptr_im[idx] = v_re * sin_t + v_im * cos_t


fn mc_transform_simd(
    mut state: QuantumState,
    controls: List[Int],
    target: Int,
    gate: Gate,
):
    """Single-threaded SIMD multi-controlled gate (mask check per index)."""
    if len(controls) == 0:
        transform_simd(state, target, gate)
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

    for k in range(0, l, 2 * stride):
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


fn c_transform_p_simd(
    mut state: QuantumState,
    control: Int,
    target: Int,
    theta: Float64,
):
    """Single-threaded SIMD controlled Phase transform."""
    var c_stride = 1 << control
    var t_stride = 1 << target
    var size = state.size()
    var ptr_re = state.re_ptr()
    var ptr_im = state.im_ptr()
    var cos_t = cos(theta)
    var sin_t = sin(theta)

    if target < control:
        var total_work = (size // (2 * c_stride)) * (c_stride // (2 * t_stride))

        for s in range(total_work):
            var segments_per_block = c_stride // (2 * t_stride)
            var k = s // segments_per_block
            var j = s % segments_per_block
            var sub_start = k * 2 * c_stride + c_stride + j * 2 * t_stride

            @parameter
            fn vectorize_p_low[width: Int](m: Int):
                var idx = sub_start + m + t_stride
                var v_re = ptr_re.load[width=width](idx)
                var v_im = ptr_im.load[width=width](idx)
                var res_re = v_re * cos_t - v_im * sin_t
                var res_im = v_re * sin_t + v_im * cos_t
                ptr_re.store[width=width](idx, res_re)
                ptr_im.store[width=width](idx, res_im)

            if t_stride >= simd_width:
                vectorize[vectorize_p_low, simd_width](t_stride)
            else:
                for m in range(t_stride):
                    var idx = sub_start + m + t_stride
                    var v_re = ptr_re[idx]
                    var v_im = ptr_im[idx]
                    ptr_re[idx] = v_re * cos_t - v_im * sin_t
                    ptr_im[idx] = v_re * sin_t + v_im * cos_t
    else:
        var total_work = (size // (2 * t_stride)) * (t_stride // (2 * c_stride))

        for s in range(total_work):
            var segments_per_block = t_stride // (2 * c_stride)
            var k = s // segments_per_block
            var p = s % segments_per_block
            var p_start = k * 2 * t_stride + p * 2 * c_stride + c_stride

            @parameter
            fn vectorize_p_high[width: Int](m: Int):
                var idx = p_start + m + t_stride
                var v_re = ptr_re.load[width=width](idx)
                var v_im = ptr_im.load[width=width](idx)
                var res_re = v_re * cos_t - v_im * sin_t
                var res_im = v_re * sin_t + v_im * cos_t
                ptr_re.store[width=width](idx, res_re)
                ptr_im.store[width=width](idx, res_im)

            if c_stride >= simd_width:
                vectorize[vectorize_p_high, simd_width](c_stride)
            else:
                for m in range(c_stride):
                    var idx = p_start + m + t_stride
                    var v_re = ptr_re[idx]
                    var v_im = ptr_im[idx]
                    ptr_re[idx] = v_re * cos_t - v_im * sin_t
                    ptr_im[idx] = v_re * sin_t + v_im * cos_t


fn c_transform_simd(
    mut state: QuantumState,
    control: Int,
    target: Int,
    gate: Gate,
    gate_kind: Int = -1,
    gate_arg: FloatType = 0,
    ctx: ExecContext = ExecContext(),
):
    """Generic SIMD controlled single-qubit transform using a control mask."""
    if gate_kind == GateKind.P and ctx.simd_use_specialized_cp:
        c_transform_p_simd(state, control, target, Float64(gate_arg))
        return
    if gate_kind == GateKind.X and ctx.simd_use_specialized_cx:
        c_transform_x_simd(state, control, target)
        return
    if gate_kind == GateKind.RY and ctx.simd_use_specialized_cry:
        c_transform_ry_simd(state, control, target, Float64(gate_arg))
        return

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

    for k in range(0, l, 2 * stride):

        @parameter
        fn vectorize_gate[width: Int](m: Int):
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

            var out_u_re = u_re * g00_re - u_im * g00_im + v_re * g01_re - v_im * g01_im
            var out_u_im = u_re * g00_im + u_im * g00_re + v_re * g01_im + v_im * g01_re
            var out_v_re = u_re * g10_re - u_im * g10_im + v_re * g11_re - v_im * g11_im
            var out_v_im = u_re * g10_im + u_im * g10_re + v_re * g11_im + v_im * g11_re

            out_u_re = mask.select(out_u_re, u_re)
            out_u_im = mask.select(out_u_im, u_im)
            out_v_re = mask.select(out_v_re, v_re)
            out_v_im = mask.select(out_v_im, v_im)

            ptr_re.store[width=width](idx, out_u_re)
            ptr_im.store[width=width](idx, out_u_im)
            ptr_re.store[width=width](idx + stride, out_v_re)
            ptr_im.store[width=width](idx + stride, out_v_im)

        if stride >= simd_width:
            vectorize[vectorize_gate, simd_width](stride)
        else:
            for m in range(stride):
                var idx = k + m
                if (idx & (1 << control)) != 0:
                    var u_re = ptr_re[idx]
                    var u_im = ptr_im[idx]
                    var v_re = ptr_re[idx + stride]
                    var v_im = ptr_im[idx + stride]

                    ptr_re[idx] = u_re * g00_re - u_im * g00_im + v_re * g01_re - v_im * g01_im
                    ptr_im[idx] = u_re * g00_im + u_im * g00_re + v_re * g01_im + v_im * g01_re
                    ptr_re[idx + stride] = u_re * g10_re - u_im * g10_im + v_re * g11_re - v_im * g11_im
                    ptr_im[idx + stride] = u_re * g10_im + u_im * g10_re + v_re * g11_im + v_im * g11_re


fn c_transform_x_simd(mut state: QuantumState, control: Int, target: Int):
    """Single-threaded SIMD controlled X transform."""
    var l = state.size()
    var stride = 1 << target
    var ptr_re = state.re_ptr()
    var ptr_im = state.im_ptr()

    for k in range(0, l, 2 * stride):

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


fn c_transform_ry_simd(
    mut state: QuantumState,
    control: Int,
    target: Int,
    theta: Float64,
):
    """Single-threaded SIMD controlled RY transform."""
    var l = state.size()
    var stride = 1 << target
    var ptr_re = state.re_ptr()
    var ptr_im = state.im_ptr()
    var cos_t = cos(theta / 2)
    var sin_t = sin(theta / 2)

    for k in range(0, l, 2 * stride):

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
