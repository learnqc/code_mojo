from collections import List

from butterfly.core.circuit import (
    Circuit,
    ControlKind,
    ControlledUnitaryTransformation,
    FusedPairTransformation,
    GateTransformation,
    Transformation,
    UnitaryTransformation,
)
from butterfly.core.gates import GateInfo, GateKind
from butterfly.core.types import Amplitude, Gate, FloatType, `0`, `1`
from butterfly.core.state import QuantumState, simd_width
from butterfly.core.transformations_unitary import apply_unitary
from butterfly.core.fused_kernels_sparse_global import (
    transform_fused_hh_sparse,
    transform_fused_hp_sparse,
    transform_fused_pp_sparse,
    transform_fused_shared_c_pp_sparse,
)


fn mul_gate(a: Gate, b: Gate) -> Gate:
    var out: Gate = [[`0`, `0`], [`0`, `0`]]
    for i in range(2):
        for j in range(2):
            var sum = `0`
            for k in range(2):
                sum = sum + a[i][k] * b[k][j]
            out[i][j] = sum
    return out^


fn gate_to_unitary(gate: Gate) -> List[Amplitude]:
    var u = List[Amplitude](capacity=4)
    u.append(gate[0][0])
    u.append(gate[0][1])
    u.append(gate[1][0])
    u.append(gate[1][1])
    return u^

fn identity_matrix4x4() -> List[Amplitude]:
    var m = List[Amplitude](length=16, fill=`0`)
    m[0] = `1`
    m[5] = `1`
    m[10] = `1`
    m[15] = `1`
    return m^

fn gate_to_matrix4x4(gate: Gate, q_low: Int, target: Int) -> List[Amplitude]:
    var m = List[Amplitude](length=16, fill=`0`)
    if target == q_low:
        for b1 in range(2):
            for row0 in range(2):
                for col0 in range(2):
                    var row = row0 + 2 * b1
                    var col = col0 + 2 * b1
                    m[row * 4 + col] = gate[row0][col0]
    else:
        for b0 in range(2):
            for row1 in range(2):
                for col1 in range(2):
                    var row = b0 + 2 * row1
                    var col = b0 + 2 * col1
                    m[row * 4 + col] = gate[row1][col1]
    return m^

fn controlled_gate_to_matrix4x4(
    gate: Gate,
    q_low: Int,
    control: Int,
    target: Int,
) -> List[Amplitude]:
    var m = List[Amplitude](length=16, fill=`0`)
    for b0 in range(2):
        for b1 in range(2):
            var in_idx = b0 + 2 * b1
            var ctrl_bit = b0 if control == q_low else b1
            var tgt_bit = b0 if target == q_low else b1
            if ctrl_bit == 0:
                m[in_idx * 4 + in_idx] = `1`
            else:
                for out_t in range(2):
                    var out_b0 = b0
                    var out_b1 = b1
                    if target == q_low:
                        out_b0 = out_t
                    else:
                        out_b1 = out_t
                    var out_idx = out_b0 + 2 * out_b1
                    m[out_idx * 4 + in_idx] = gate[out_t][tgt_bit]
    return m^

fn matmul_4x4(a: List[Amplitude], b: List[Amplitude]) -> List[Amplitude]:
    var out = List[Amplitude](length=16, fill=`0`)
    for row in range(4):
        for col in range(4):
            var sum = `0`
            for k in range(4):
                sum = sum + a[row * 4 + k] * b[k * 4 + col]
            out[row * 4 + col] = sum
    return out^

fn gate_bounds(
    target: Int,
    control_kind: Int,
    control: Int,
) -> Tuple[Int, Int]:
    var low = target
    var high = target
    if control_kind == ControlKind.SINGLE_CONTROL:
        if control < low:
            low = control
        if control > high:
            high = control
    return (low, high)

fn is_special_pair_gate(gate_tr: GateTransformation) -> Bool:
    if gate_tr.kind == ControlKind.NO_CONTROL:
        return (
            gate_tr.gate_info.kind == GateKind.H
            or gate_tr.gate_info.kind == GateKind.P
        )
    if gate_tr.kind == ControlKind.SINGLE_CONTROL:
        return gate_tr.gate_info.kind == GateKind.P
    return False

fn try_pair_matrix(
    gate_tr: GateTransformation,
    q_low: Int,
    q_high: Int,
) -> Tuple[Bool, List[Amplitude]]:
    if not is_special_pair_gate(gate_tr):
        return (False, List[Amplitude]())
    if gate_tr.kind == ControlKind.NO_CONTROL:
        if gate_tr.target != q_low and gate_tr.target != q_high:
            return (False, List[Amplitude]())
        return (
            True,
            gate_to_matrix4x4(gate_tr.gate_info.gate, q_low, gate_tr.target),
        )
    if gate_tr.kind == ControlKind.SINGLE_CONTROL:
        var control = gate_tr.controls[0]
        var target = gate_tr.target
        if control == target:
            return (False, List[Amplitude]())
        if (control != q_low and control != q_high) or (
            target != q_low and target != q_high
        ):
            return (False, List[Amplitude]())
        return (
            True,
            controlled_gate_to_matrix4x4(
                gate_tr.gate_info.gate,
                q_low,
                control,
                target,
            ),
        )
    return (False, List[Amplitude]())

fn is_h_gate(gate_tr: GateTransformation) -> Bool:
    return (
        gate_tr.kind == ControlKind.NO_CONTROL
        and gate_tr.gate_info.kind == GateKind.H
    )

fn is_p_gate(gate_tr: GateTransformation) -> Bool:
    return (
        gate_tr.kind == ControlKind.NO_CONTROL
        and gate_tr.gate_info.kind == GateKind.P
    )

fn is_cp_gate(gate_tr: GateTransformation) -> Bool:
    return (
        gate_tr.kind == ControlKind.SINGLE_CONTROL
        and gate_tr.gate_info.kind == GateKind.P
    )

fn can_fuse_special_pair(t0: GateTransformation, t1: GateTransformation) -> Bool:
    if t0.kind == ControlKind.MULTI_CONTROL or t1.kind == ControlKind.MULTI_CONTROL:
        return False
    if (
        is_cp_gate(t0)
        and is_cp_gate(t1)
        and t0.controls[0] == t1.controls[0]
        and t0.target != t1.target
    ):
        return True
    var low = t0.target
    var high = t0.target
    if t0.kind == ControlKind.SINGLE_CONTROL:
        var c0 = t0.controls[0]
        if c0 < low:
            low = c0
        if c0 > high:
            high = c0
    if t1.kind == ControlKind.SINGLE_CONTROL:
        var c1 = t1.controls[0]
        if c1 < low:
            low = c1
        if c1 > high:
            high = c1
    if t1.target < low:
        low = t1.target
    if t1.target > high:
        high = t1.target
    if high - low != 1:
        return False
    if is_h_gate(t0) and is_h_gate(t1):
        return True
    if is_p_gate(t0) and is_p_gate(t1):
        return True
    if (is_h_gate(t0) and is_cp_gate(t1)) or (
        is_cp_gate(t0) and is_h_gate(t1)
    ):
        return True
    return False

fn apply_fused_pair(mut state: QuantumState, pair: FusedPairTransformation) raises:
    var t0 = pair.first.copy()
    var t1 = pair.second.copy()

    if is_cp_gate(t0) and is_cp_gate(t1) and t0.controls[0] == t1.controls[0]:
        var arg0 = Float64(t0.gate_info.arg.value())
        var arg1 = Float64(t1.gate_info.arg.value())
        if t0.target > t1.target:
            transform_fused_shared_c_pp_sparse[simd_width](
                state,
                t0.controls[0],
                t0.target,
                t1.target,
                arg0,
                arg1,
            )
        else:
            transform_fused_shared_c_pp_sparse[simd_width](
                state,
                t0.controls[0],
                t1.target,
                t0.target,
                arg1,
                arg0,
            )
        return

    if is_h_gate(t0) and is_h_gate(t1):
        transform_fused_hh_sparse[simd_width](state, t0.target, t1.target)
        return

    if is_p_gate(t0) and is_p_gate(t1):
        var arg0 = Float64(t0.gate_info.arg.value())
        var arg1 = Float64(t1.gate_info.arg.value())
        if t0.target > t1.target:
            transform_fused_pp_sparse[simd_width](
                state, t0.target, t1.target, arg0, arg1
            )
        else:
            transform_fused_pp_sparse[simd_width](
                state, t1.target, t0.target, arg1, arg0
            )
        return

    if is_h_gate(t0) and is_p_gate(t1):
        transform_fused_hp_sparse[simd_width](
            state, t0.target, t1.target, Float64(t1.gate_info.arg.value())
        )
        return

    if is_p_gate(t0) and is_h_gate(t1):
        transform_fused_hp_sparse[simd_width](
            state, t1.target, t0.target, Float64(t0.gate_info.arg.value())
        )
        return

    var control0 = 0
    if t0.kind == ControlKind.SINGLE_CONTROL:
        control0 = t0.controls[0]
    var control1 = 0
    if t1.kind == ControlKind.SINGLE_CONTROL:
        control1 = t1.controls[0]
    var bounds0 = gate_bounds(t0.target, t0.kind, control0)
    var bounds1: Tuple[Int, Int] = gate_bounds(t1.target, t1.kind, control1)
    var q_low = bounds0[0]
    if bounds1[0] < q_low:
        q_low = bounds1[0]
    # var q_high: Int = bounds0[1]
    # if bounds1[1] > q_high:
    #     q_high = bounds1[1]

    var m0 = gate_to_matrix4x4(t0.gate_info.gate, q_low, t0.target)
    if t0.kind == ControlKind.SINGLE_CONTROL:
        m0 = controlled_gate_to_matrix4x4(
            t0.gate_info.gate,
            q_low,
            t0.controls[0],
            t0.target,
        )
    var m1 = gate_to_matrix4x4(t1.gate_info.gate, q_low, t1.target)
    if t1.kind == ControlKind.SINGLE_CONTROL:
        m1 = controlled_gate_to_matrix4x4(
            t1.gate_info.gate,
            q_low,
            t1.controls[0],
            t1.target,
        )

    var m = matmul_4x4(m1, m0)
    apply_unitary(state, m, q_low, 2)

fn fuse_special_pair_gates[StateType: AnyType](
    transformations: List[Transformation[StateType]],
) -> List[Transformation[StateType]]:
    var fused = List[Transformation[StateType]]()
    var i = 0
    var n = len(transformations)
    while i < n:
        var tr = transformations[i]
        if (
            i + 1 < n
            and tr.isa[GateTransformation]()
            and transformations[i + 1].isa[GateTransformation]()
        ):
            var t0 = tr[GateTransformation].copy()
            var t1 = transformations[i + 1][GateTransformation].copy()
            if can_fuse_special_pair(t0, t1):
                fused.append(FusedPairTransformation(t0, t1))
                i += 2
                continue
        fused.append(tr.copy())
        i += 1
    return fused^

fn flush_active[StateType: AnyType](
    mut fused: List[Transformation[StateType]],
    mut active: Bool,
    mut active_count: Int,
    active_gate: Gate,
    active_target: Int,
    active_control_kind: Int,
    active_control: Int,
    active_gate_info: GateInfo,
    active_is_phase: Bool,
    active_phase_sum: FloatType,
) raises:
    if not active:
        return
    if active_is_phase:
        var controls = List[Int]()
        if active_control_kind == ControlKind.SINGLE_CONTROL:
            controls.append(active_control)
        fused.append(
            GateTransformation(
                controls,
                active_target,
                GateInfo(GateKind.P, active_phase_sum),
            )
        )
    elif active_count == 1:
        var controls = List[Int]()
        if active_control_kind == ControlKind.SINGLE_CONTROL:
            controls.append(active_control)
        fused.append(
            GateTransformation(
                controls,
                active_target,
                active_gate_info.copy(),
            )
        )
    else:
        var u = gate_to_unitary(active_gate)
        if active_control_kind == ControlKind.NO_CONTROL:
            fused.append(
                UnitaryTransformation(u^, active_target, 1, "fused")
            )
        else:
            fused.append(
                ControlledUnitaryTransformation(
                    u^,
                    active_target,
                    active_control,
                    1,
                    "fused_ctrl",
                )
            )
    active = False
    active_count = 0

fn flush_pair[StateType: AnyType](
    mut fused: List[Transformation[StateType]],
    mut mode: Int,
    mut active_count: Int,
    active_matrix: List[Amplitude],
    active_pair_low: Int,
    active_gate_info: GateInfo,
    active_target: Int,
    active_control_kind: Int,
    active_control: Int,
) raises:
    if mode == 0:
        return
    if mode == 1:
        var controls = List[Int]()
        if active_control_kind == ControlKind.SINGLE_CONTROL:
            controls.append(active_control)
        fused.append(
            GateTransformation(
                controls,
                active_target,
                active_gate_info.copy(),
            )
        )
    else:
        fused.append(
            UnitaryTransformation(
                active_matrix.copy(),
                active_pair_low,
                2,
                "fused_pair",
            )
        )
    mode = 0
    active_count = 0


fn fuse_same_target_gates[StateType: AnyType](
    transformations: List[Transformation[StateType]],
) raises -> List[Transformation[StateType]]:
    var fused = List[Transformation[StateType]]()
    var active = False
    var active_gate: Gate = [[`1`, `0`], [`0`, `1`]]
    var active_count = 0
    var active_target = 0
    var active_control_kind = ControlKind.NO_CONTROL
    var active_control = 0
    var active_gate_info = GateInfo(0)
    var active_is_phase = False
    var active_phase_sum = FloatType(0)

    for tr in transformations:
        if tr.isa[GateTransformation]():
            var gate_tr = tr[GateTransformation].copy()
            var control_kind = gate_tr.kind
            if control_kind == ControlKind.MULTI_CONTROL:
                flush_active(
                    fused,
                    active,
                    active_count,
                    active_gate,
                    active_target,
                    active_control_kind,
                    active_control,
                    active_gate_info,
                    active_is_phase,
                    active_phase_sum,
                )
                fused.append(tr.copy())
                continue

            var control = 0
            if control_kind == ControlKind.SINGLE_CONTROL:
                control = gate_tr.controls[0]

            if (
                active
                and gate_tr.target == active_target
                and control_kind == active_control_kind
                and control == active_control
            ):
                active_gate = mul_gate(gate_tr.gate_info.gate, active_gate)
                active_count += 1
                if (
                    active_is_phase
                    and gate_tr.gate_info.kind == GateKind.P
                    and gate_tr.gate_info.arg
                ):
                    active_phase_sum += gate_tr.gate_info.arg.value()
                else:
                    active_is_phase = False
                continue

            flush_active(
                fused,
                active,
                active_count,
                active_gate,
                active_target,
                active_control_kind,
                active_control,
                active_gate_info,
                active_is_phase,
                active_phase_sum,
            )
            active = True
            active_gate = gate_tr.gate_info.gate
            active_count = 1
            active_target = gate_tr.target
            active_control_kind = control_kind
            active_control = control
            active_gate_info = gate_tr.gate_info.copy()
            active_is_phase = (
                gate_tr.gate_info.kind == GateKind.P
                and gate_tr.gate_info.arg
            )
            if active_is_phase:
                active_phase_sum = gate_tr.gate_info.arg.value()
            else:
                active_phase_sum = FloatType(0)
        else:
            flush_active(
                fused,
                active,
                active_count,
                active_gate,
                active_target,
                active_control_kind,
                active_control,
                active_gate_info,
                active_is_phase,
                active_phase_sum,
            )
            fused.append(tr.copy())

    flush_active(
        fused,
        active,
        active_count,
        active_gate,
        active_target,
        active_control_kind,
        active_control,
        active_gate_info,
        active_is_phase,
        active_phase_sum,
    )
    return fused^

fn fuse_contiguous_target_pairs[StateType: AnyType](
    transformations: List[Transformation[StateType]],
) raises -> List[Transformation[StateType]]:
    var fused = List[Transformation[StateType]]()
    var mode = 0  # 0=idle, 1=pending gate, 2=active pair
    var active_count = 0
    var active_matrix = identity_matrix4x4()
    var active_pair_low = 0
    var active_pair_high = 0
    var active_gate_info = GateInfo(0)
    var active_target = 0

    for tr in transformations:
        if tr.isa[GateTransformation]():
            var gate_tr = tr[GateTransformation].copy()
            if gate_tr.kind != ControlKind.NO_CONTROL:
                flush_pair(
                    fused,
                    mode,
                    active_count,
                    active_matrix,
                    active_pair_low,
                    active_gate_info,
                    active_target,
                    ControlKind.NO_CONTROL,
                    0,
                )
                fused.append(tr.copy())
                continue

            var target = gate_tr.target
            if mode == 0:
                mode = 1
                active_count = 1
                active_gate_info = gate_tr.gate_info.copy()
                active_target = target
                continue

            if mode == 1:
                if target == active_target:
                    flush_pair(
                        fused,
                        mode,
                        active_count,
                        active_matrix,
                        active_pair_low,
                        active_gate_info,
                        active_target,
                        ControlKind.NO_CONTROL,
                        0,
                    )
                    mode = 1
                    active_count = 1
                    active_gate_info = gate_tr.gate_info.copy()
                    active_target = target
                    continue

                if abs(target - active_target) == 1:
                    active_pair_low = min(active_target, target)
                    active_pair_high = max(active_target, target)
                    var first_matrix = gate_to_matrix4x4(
                        active_gate_info.gate,
                        active_pair_low,
                        active_target,
                    )
                    var next_matrix = gate_to_matrix4x4(
                        gate_tr.gate_info.gate,
                        active_pair_low,
                        target,
                    )
                    active_matrix = matmul_4x4(next_matrix, first_matrix)
                    mode = 2
                    active_count = 2
                    continue

                flush_pair(
                    fused,
                    mode,
                    active_count,
                    active_matrix,
                    active_pair_low,
                    active_gate_info,
                    active_target,
                    ControlKind.NO_CONTROL,
                    0,
                )
                mode = 1
                active_count = 1
                active_gate_info = gate_tr.gate_info.copy()
                active_target = target
                continue

            if mode == 2:
                if target == active_pair_low or target == active_pair_high:
                    var next_matrix = gate_to_matrix4x4(
                        gate_tr.gate_info.gate,
                        active_pair_low,
                        target,
                    )
                    active_matrix = matmul_4x4(next_matrix, active_matrix)
                    active_count += 1
                    continue

                flush_pair(
                    fused,
                    mode,
                    active_count,
                    active_matrix,
                    active_pair_low,
                    active_gate_info,
                    active_target,
                    ControlKind.NO_CONTROL,
                    0,
                )
                mode = 1
                active_count = 1
                active_gate_info = gate_tr.gate_info.copy()
                active_target = target
                continue
        else:
            flush_pair(
                fused,
                mode,
                active_count,
                active_matrix,
                active_pair_low,
                active_gate_info,
                active_target,
                ControlKind.NO_CONTROL,
                0,
            )
            fused.append(tr.copy())

    flush_pair(
        fused,
        mode,
        active_count,
        active_matrix,
        active_pair_low,
        active_gate_info,
        active_target,
        ControlKind.NO_CONTROL,
        0,
    )
    return fused^

fn fuse_contiguous_target_pairs_specialized[StateType: AnyType](
    transformations: List[Transformation[StateType]],
) raises -> List[Transformation[StateType]]:
    var fused = List[Transformation[StateType]]()
    var mode = 0  # 0=idle, 1=pending gate, 2=active pair
    var active_count = 0
    var active_matrix = identity_matrix4x4()
    var active_pair_low = 0
    var active_pair_high = 0
    var active_gate_info = GateInfo(0)
    var active_target = 0
    var active_control_kind = ControlKind.NO_CONTROL
    var active_control = 0

    for tr in transformations:
        if tr.isa[GateTransformation]():
            var gate_tr = tr[GateTransformation].copy()
            if gate_tr.kind == ControlKind.MULTI_CONTROL:
                flush_pair(
                    fused,
                    mode,
                    active_count,
                    active_matrix,
                    active_pair_low,
                    active_gate_info,
                    active_target,
                    active_control_kind,
                    active_control,
                )
                fused.append(tr.copy())
                continue

            if not is_special_pair_gate(gate_tr):
                flush_pair(
                    fused,
                    mode,
                    active_count,
                    active_matrix,
                    active_pair_low,
                    active_gate_info,
                    active_target,
                    active_control_kind,
                    active_control,
                )
                fused.append(tr.copy())
                continue

            var target = gate_tr.target
            var control = 0
            if gate_tr.kind == ControlKind.SINGLE_CONTROL:
                control = gate_tr.controls[0]

            if mode == 0:
                mode = 1
                active_count = 1
                active_gate_info = gate_tr.gate_info.copy()
                active_target = target
                active_control_kind = gate_tr.kind
                active_control = control
                continue

            if mode == 1:
                if (
                    target == active_target
                    and gate_tr.kind == active_control_kind
                    and control == active_control
                ):
                    flush_pair(
                        fused,
                        mode,
                        active_count,
                        active_matrix,
                        active_pair_low,
                        active_gate_info,
                        active_target,
                        active_control_kind,
                        active_control,
                    )
                    mode = 1
                    active_count = 1
                    active_gate_info = gate_tr.gate_info.copy()
                    active_target = target
                    active_control_kind = gate_tr.kind
                    active_control = control
                    continue

                var bounds0 = gate_bounds(
                    active_target,
                    active_control_kind,
                    active_control,
                )
                var bounds1 = gate_bounds(target, gate_tr.kind, control)
                var q_low = bounds0[0]
                if bounds1[0] < q_low:
                    q_low = bounds1[0]
                var q_high = bounds0[1]
                if bounds1[1] > q_high:
                    q_high = bounds1[1]

                if q_high - q_low == 1:
                    var controls0 = List[Int]()
                    if active_control_kind == ControlKind.SINGLE_CONTROL:
                        controls0.append(active_control)
                    var active_tr = GateTransformation(
                        controls0,
                        active_target,
                        active_gate_info.copy(),
                    )
                    var res0 = try_pair_matrix(active_tr, q_low, q_high)
                    var res1 = try_pair_matrix(gate_tr, q_low, q_high)
                    if res0[0] and res1[0]:
                        active_pair_low = q_low
                        active_pair_high = q_high
                        active_matrix = matmul_4x4(res1[1], res0[1])
                        mode = 2
                        active_count = 2
                        continue

                flush_pair(
                    fused,
                    mode,
                    active_count,
                    active_matrix,
                    active_pair_low,
                    active_gate_info,
                    active_target,
                    active_control_kind,
                    active_control,
                )
                mode = 1
                active_count = 1
                active_gate_info = gate_tr.gate_info.copy()
                active_target = target
                active_control_kind = gate_tr.kind
                active_control = control
                continue

            if mode == 2:
                var res = try_pair_matrix(
                    gate_tr,
                    active_pair_low,
                    active_pair_high,
                )
                if res[0]:
                    active_matrix = matmul_4x4(res[1], active_matrix)
                    active_count += 1
                    continue

                flush_pair(
                    fused,
                    mode,
                    active_count,
                    active_matrix,
                    active_pair_low,
                    active_gate_info,
                    active_target,
                    active_control_kind,
                    active_control,
                )
                mode = 1
                active_count = 1
                active_gate_info = gate_tr.gate_info.copy()
                active_target = target
                active_control_kind = gate_tr.kind
                active_control = control
                continue
        else:
            flush_pair(
                fused,
                mode,
                active_count,
                active_matrix,
                active_pair_low,
                active_gate_info,
                active_target,
                active_control_kind,
                active_control,
            )
            fused.append(tr.copy())

    flush_pair(
        fused,
        mode,
        active_count,
        active_matrix,
        active_pair_low,
        active_gate_info,
        active_target,
        active_control_kind,
        active_control,
    )
    return fused^


fn fuse_transformations[StateType: AnyType](
    transformations: List[Transformation[StateType]],
) raises -> List[Transformation[StateType]]:
    var fused_pairs = fuse_contiguous_target_pairs(transformations)
    return fuse_same_target_gates(fused_pairs)

fn fuse_transformations_specialized_pairs[StateType: AnyType](
    transformations: List[Transformation[StateType]],
) raises -> List[Transformation[StateType]]:
    var fused_pairs = fuse_special_pair_gates(transformations)
    return fuse_same_target_gates(fused_pairs)


fn fuse_circuit[StateType: AnyType](
    circuit: Circuit[StateType],
) raises -> Circuit[StateType]:
    var fused = Circuit[StateType](circuit.num_qubits)
    fused.registers = circuit.registers.copy()
    fused.transformations = fuse_transformations(circuit.transformations.copy())
    return fused^

fn fuse_circuit_specialized_pairs[StateType: AnyType](
    circuit: Circuit[StateType],
) raises -> Circuit[StateType]:
    var fused = Circuit[StateType](circuit.num_qubits)
    fused.registers = circuit.registers.copy()
    fused.transformations = fuse_transformations_specialized_pairs(
        circuit.transformations.copy()
    )
    return fused^
