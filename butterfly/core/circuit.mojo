from butterfly.core.state import (
    QuantumState,
    transform,
    c_transform,
    mc_transform_interval,
    transform_u,
    c_transform_u,
    bit_reverse_state,
    apply_cft_stage,
    partial_bit_reverse_state,
)
from butterfly.core.execute_simd_dispatch import (
    execute_transformations_simd,
)
from butterfly.core.execute_simd_v2_dispatch import (
    execute_transformations_simd_v2,
)
from butterfly.core.execution_strategy import (
    ExecutionStrategy,
    Generic,
    SIMD,
    SIMDv2,
    FusedV3,
    GENERIC,
    SIMD_STRATEGY,
    SIMD_V2,
    FUSED_V3,
)
from butterfly.core.types import (
    Amplitude,
    Gate,
    FloatType,
    pi,
    # Matrix2x2,
    # ComplexMatrix,
)
from butterfly.core.gates import *
from butterfly.algos.fused_gates import (
    transform_matrix4,
    transform_matrix8,
    transform_matrix16,
    compute_kron_product,
)
from butterfly.algos.unitary_kernels import (
    Matrix4x4,
    Matrix8x8,
    Matrix16x16,
    matmul_matrix4x4,
    matmul_matrix8x8,
    matmul_matrix16x16,
)
from collections import InlineArray
from butterfly.core.gates import *


from utils.variant import Variant


struct GateTransformation(Copyable, Movable):
    var gate: Gate
    var target: Int
    var name: String
    var arg: FloatType

    fn __init__(
        out self,
        gate: Gate,
        target: Int,
        name: String = "unitary",
        arg: FloatType = 0.0,
    ):
        self.gate = gate
        self.target = target
        self.name = name
        self.arg = arg

    fn __copyinit__(out self, existing: Self):
        self.gate = existing.gate
        self.target = existing.target
        self.name = existing.name
        self.arg = existing.arg

    fn __moveinit__(out self, deinit existing: Self):
        self.gate = existing.gate
        self.target = existing.target
        self.name = existing.name^
        self.arg = existing.arg


struct SingleControlGateTransformation(Copyable, Movable):
    var gate: Gate
    var target: Int
    var control: Int
    var name: String
    var arg: FloatType

    fn __init__(
        out self,
        gate: Gate,
        target: Int,
        control: Int,
        name: String = "unitary",
        arg: FloatType = 0.0,
    ):
        self.gate = gate
        self.target = target
        self.control = control
        self.name = name
        self.arg = arg

    fn __copyinit__(out self, existing: Self):
        self.gate = existing.gate
        self.target = existing.target
        self.control = existing.control
        self.name = existing.name
        self.arg = existing.arg

    fn __moveinit__(out self, deinit existing: Self):
        self.gate = existing.gate
        self.target = existing.target
        self.control = existing.control
        self.name = existing.name^
        self.arg = existing.arg


struct MultiControlGateTransformation(Copyable, Movable):
    var gate: Gate
    var target: Int
    var controls: List[Int]
    var name: String
    var arg: FloatType

    fn __init__(
        out self,
        gate: Gate,
        target: Int,
        controls: List[Int],
        name: String = "unitary",
        arg: FloatType = 0.0,
    ):
        self.gate = gate
        self.target = target
        self.controls = List[Int](capacity=len(controls))
        for i in range(len(controls)):
            self.controls.append(controls[i])
        self.name = name
        self.arg = arg

    fn __copyinit__(out self, existing: Self):
        self.gate = existing.gate
        self.target = existing.target
        self.controls = List[Int](capacity=len(existing.controls))
        for i in range(len(existing.controls)):
            self.controls.append(existing.controls[i])
        self.name = existing.name
        self.arg = existing.arg

    fn __moveinit__(out self, deinit existing: Self):
        self.gate = existing.gate
        self.target = existing.target
        self.controls = existing.controls^
        self.name = existing.name^
        self.arg = existing.arg


struct UnitaryTransformation(Copyable, Movable):
    var u: List[Amplitude]
    var target: Int
    var m: Int
    var name: String

    fn __init__(
        out self,
        deinit u: List[Amplitude],
        target: Int,
        m: Int,
        name: String = "unitary",
    ):
        self.u = u^
        self.target = target
        self.m = m
        self.name = name

    fn __copyinit__(out self, existing: Self):
        self.u = existing.u.copy()
        self.target = existing.target
        self.m = existing.m
        self.name = existing.name

    fn __moveinit__(out self, deinit existing: Self):
        self.u = existing.u^
        self.target = existing.target
        self.m = existing.m
        self.name = existing.name^


struct ControlledUnitaryTransformation(Copyable, Movable):
    var u: List[Amplitude]
    var target: Int
    var control: Int
    var m: Int
    var name: String

    fn __init__(
        out self,
        deinit u: List[Amplitude],
        target: Int,
        control: Int,
        m: Int,
        name: String = "unitary",
    ):
        self.u = u^
        self.target = target
        self.control = control
        self.m = m
        self.name = name

    fn __copyinit__(out self, existing: Self):
        self.u = existing.u.copy()
        self.target = existing.target
        self.control = existing.control
        self.m = existing.m
        self.name = existing.name


struct BitReversalTransformation(Copyable, Movable):
    fn __init__(out self):
        pass

    fn __copyinit__(out self, existing: Self):
        pass

    fn __moveinit__(out self, deinit existing: Self):
        pass


struct DiagonalTransformation(Copyable, Movable):
    var items: List[Int]
    var offset: Int
    var size: Int

    fn __init__(out self, var items: List[Int], offset: Int = 0, size: Int = 0):
        self.items = items^
        self.offset = offset
        self.size = size

    fn __copyinit__(out self, existing: Self):
        self.items = existing.items.copy()
        self.offset = existing.offset
        self.size = existing.size

    fn __moveinit__(out self, deinit existing: Self):
        self.items = existing.items^
        self.offset = existing.offset
        self.size = existing.size


alias Transformation = Variant[
    GateTransformation,
    SingleControlGateTransformation,
    MultiControlGateTransformation,
    BitReversalTransformation,
    UnitaryTransformation,
    ControlledUnitaryTransformation,
    DiagonalTransformation,
]


fn is_permutation(t: Transformation) -> Bool:
    return t.isa[BitReversalTransformation]()


fn get_involved_qubits(t: Transformation) -> List[Int]:
    var res = List[Int]()
    if t.isa[GateTransformation]():
        res.append(t[GateTransformation].copy().target)
    elif t.isa[SingleControlGateTransformation]():
        var g = t[SingleControlGateTransformation].copy()
        res.append(g.target)
        res.append(g.control)
    elif t.isa[MultiControlGateTransformation]():
        var g = t[MultiControlGateTransformation].copy()
        res.append(g.target)
        for i in range(len(g.controls)):
            res.append(g.controls[i])
    elif t.isa[UnitaryTransformation]():
        var g = t[UnitaryTransformation].copy()
        for i in range(g.m):
            res.append(g.target + i)
    elif t.isa[ControlledUnitaryTransformation]():
        var g = t[ControlledUnitaryTransformation].copy()
        res.append(g.control)
        for i in range(g.m):
            res.append(g.target + i)
    return res^


fn is_controlled(t: Transformation) -> Bool:
    if (
        t.isa[SingleControlGateTransformation]()
        or t.isa[MultiControlGateTransformation]()
    ):
        return True
    return False


fn num_controls(t: Transformation) -> Int:
    if t.isa[SingleControlGateTransformation]():
        return 1
    elif t.isa[MultiControlGateTransformation]():
        return len(t[MultiControlGateTransformation].copy().controls)
    return 0


fn get_target(t: Transformation) -> Int:
    if t.isa[GateTransformation]():
        return t[GateTransformation].target
    elif t.isa[SingleControlGateTransformation]():
        return t[SingleControlGateTransformation].target
    elif t.isa[MultiControlGateTransformation]():
        return t[MultiControlGateTransformation].target
    elif t.isa[UnitaryTransformation]():
        return t[UnitaryTransformation].target
    elif t.isa[ControlledUnitaryTransformation]():
        return t[ControlledUnitaryTransformation].target
    return -1


fn get_gate(t: Transformation) -> Gate:
    # Assumes not permutation
    if t.isa[GateTransformation]():
        return t[GateTransformation].gate
    elif t.isa[SingleControlGateTransformation]():
        return t[SingleControlGateTransformation].gate
    elif t.isa[MultiControlGateTransformation]():
        return t[MultiControlGateTransformation].gate
    return X


fn get_controls(t: Transformation) -> List[Int]:
    var res = List[Int]()
    if t.isa[SingleControlGateTransformation]():
        res.append(t[SingleControlGateTransformation].copy().control)
    elif t.isa[MultiControlGateTransformation]():
        var g = t[MultiControlGateTransformation].copy()
        for i in range(len(g.controls)):
            res.append(g.controls[i])
    return res^


fn get_as_matrix4x4(t: Transformation, q_high: Int, q_low: Int) -> Matrix4x4:
    var identity2 = Gate(
        InlineArray[Amplitude, 2](Amplitude(1, 0), Amplitude(0, 0)),
        InlineArray[Amplitude, 2](Amplitude(0, 0), Amplitude(1, 0)),
    )

    if t.isa[BitReversalTransformation]():
        return compute_kron_product(identity2, identity2)

    # var gate = X
    # var target = -1
    var is_controlled = False
    var num_controls = 0
    var controls = List[Int]()

    if t.isa[GateTransformation]():
        var g = t[GateTransformation].copy()
        gate = g.gate
        target = g.target
        # is_controlled is already False
    elif t.isa[SingleControlGateTransformation]():
        var g = t[SingleControlGateTransformation].copy()
        gate = g.gate
        target = g.target
        is_controlled = True
        num_controls = 1
        controls.append(g.control)
    elif t.isa[MultiControlGateTransformation]():
        var g = t[MultiControlGateTransformation].copy()
        gate = g.gate
        target = g.target
        is_controlled = True
        num_controls = len(g.controls)
        # Avoid copy if possible, but safe to just copy/reference
        # We need to access controls by index later
        for i in range(len(g.controls)):
            controls.append(g.controls[i])
    else:
        # Fallback/Error
        return compute_kron_product(identity2, identity2)

    if not is_controlled:
        if target == q_high:
            return compute_kron_product(gate, identity2)
        elif target == q_low:
            return compute_kron_product(identity2, gate)
        else:
            return compute_kron_product(identity2, identity2)

    var row = InlineArray[Amplitude, 4](
        Amplitude(0, 0), Amplitude(0, 0), Amplitude(0, 0), Amplitude(0, 0)
    )
    var m = Matrix4x4(row, row, row, row)

    for i in range(4):
        var b_high = (i >> 1) & 1
        var b_low = i & 1

        var controls_satisfied = True
        for q_idx in range(num_controls):
            var ctrl = controls[q_idx]
            var val: Int
            if ctrl == q_high:
                val = b_high
            elif ctrl == q_low:
                val = b_low
            else:
                controls_satisfied = False
                break
            if val == 0:
                controls_satisfied = False
                break

        if not controls_satisfied:
            m[i][i] = Amplitude(1, 0)
        else:
            var b_t: Int
            if target == q_high:
                b_t = b_high
            else:
                b_t = b_low

            m[i][i] = gate[b_t][b_t]
            var flipped_j = i ^ (2 if target == q_high else 1)
            m[flipped_j][i] = gate[1 - b_t][b_t]

    return m


# Using simplified logic for 8x8 and 16x16 or strictly copying the logic if needed
# For brevity and safety, let's implement the helpers fully in next steps if needed,
# or adapt existing logic.
# The `fuse` function needs `get_as_matrix8x8` and `get_as_matrix16x16`.
# I will implement simplified versions or placehodlers if not critical, but fusion is critical.
# I'll implement them fully.


fn _kron4_2_helper(m4: Matrix4x4, m2: Gate) -> Matrix8x8:
    var row = InlineArray[Amplitude, 8](
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
    )
    var res = Matrix8x8(row, row, row, row, row, row, row, row)
    for i in range(4):
        for j in range(4):
            for k in range(2):
                for l in range(2):
                    res[2 * i + k][2 * j + l] = m4[i][j] * m2[k][l]
    return res


fn get_as_matrix8x8(
    t: Transformation, q_high: Int, q_mid: Int, q_low: Int
) -> Matrix8x8:
    var identity2 = Gate(
        InlineArray[Amplitude, 2](Amplitude(1, 0), Amplitude(0, 0)),
        InlineArray[Amplitude, 2](Amplitude(0, 0), Amplitude(1, 0)),
    )

    # Helper to create an 8x8 identity (omitted full unrolled for brevity, trust default init?)
    # Reusing row-based init from original
    var r8 = InlineArray[Amplitude, 8](
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
    )
    # ... Assume Matrix8x8 identity is simpler to make if supported, otherwise manual.
    # Manual init:
    var id8 = Matrix8x8(r8, r8, r8, r8, r8, r8, r8, r8)
    for i in range(8):
        id8[i][i] = Amplitude(1, 0)  # inefficient but works

    if t.isa[BitReversalTransformation]():
        return id8

    var gate = X
    var target = -1
    var is_controlled = False
    var num_controls = 0
    var controls = List[Int]()

    if t.isa[GateTransformation]():
        var g = t[GateTransformation].copy()
        gate = g.gate
        target = g.target
    elif t.isa[SingleControlGateTransformation]():
        var g = t[SingleControlGateTransformation].copy()
        gate = g.gate
        target = g.target
        is_controlled = True
        num_controls = 1
        controls.append(g.control)
    elif t.isa[MultiControlGateTransformation]():
        var g = t[MultiControlGateTransformation].copy()
        gate = g.gate
        target = g.target
        is_controlled = True
        num_controls = len(g.controls)
        for i in range(len(g.controls)):
            controls.append(g.controls[i])

    if not is_controlled:
        if target == q_high:
            var m4 = compute_kron_product(gate, identity2)
            return _kron4_2_helper(m4, identity2)
        elif target == q_mid:
            var m4 = compute_kron_product(identity2, gate)
            return _kron4_2_helper(m4, identity2)
        elif target == q_low:
            var m4 = compute_kron_product(identity2, identity2)
            return _kron4_2_helper(m4, gate)
        else:
            return id8

    var m = Matrix8x8(
        r8, r8, r8, r8, r8, r8, r8, r8
    )  # initialized to 0s? check constructor.
    # Original code initialized explicit rows for ID and 0s.
    # Constructor likely makes copies of passed rows. If passed r8 (0s), it's 0s.

    for i in range(8):
        var b_high = (i >> 2) & 1
        var b_mid = (i >> 1) & 1
        var b_low = i & 1

        var controls_satisfied = True
        for q_idx in range(num_controls):
            var ctrl = controls[q_idx]
            var val: Int
            if ctrl == q_high:
                val = b_high
            elif ctrl == q_mid:
                val = b_mid
            elif ctrl == q_low:
                val = b_low
            else:
                controls_satisfied = False
                break
            if val == 0:
                controls_satisfied = False
                break

        if not controls_satisfied:
            m[i][i] = Amplitude(1, 0)
        else:
            var b_t: Int
            if target == q_high:
                b_t = b_high
            elif target == q_mid:
                b_t = b_mid
            else:
                b_t = b_low

            m[i][i] = gate[b_t][b_t]
            var flipped_j: Int
            if target == q_high:
                flipped_j = i ^ 4
            elif target == q_mid:
                flipped_j = i ^ 2
            else:
                flipped_j = i ^ 1
            m[flipped_j][i] = gate[1 - b_t][b_t]
    return m


fn get_as_matrix16x16(
    t: Transformation, q3: Int, q2: Int, q1: Int, q0: Int
) -> Matrix16x16:
    var r16 = InlineArray[Amplitude, 16](
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
        Amplitude(0, 0),
    )
    var id16 = Matrix16x16(
        r16,
        r16,
        r16,
        r16,
        r16,
        r16,
        r16,
        r16,
        r16,
        r16,
        r16,
        r16,
        r16,
        r16,
        r16,
        r16,
    )
    for i in range(16):
        id16[i][i] = Amplitude(1, 0)

    if t.isa[BitReversalTransformation]():
        return id16

    var gate = X
    var target = -1
    var is_controlled = False
    var num_controls = 0
    var controls = List[Int]()

    if t.isa[GateTransformation]():
        var g = t[GateTransformation].copy()
        gate = g.gate
        target = g.target
    elif t.isa[SingleControlGateTransformation]():
        var g = t[SingleControlGateTransformation].copy()
        gate = g.gate
        target = g.target
        is_controlled = True
        num_controls = 1
        controls.append(g.control)
    elif t.isa[MultiControlGateTransformation]():
        var g = t[MultiControlGateTransformation].copy()
        gate = g.gate
        target = g.target
        is_controlled = True
        num_controls = len(g.controls)
        for i in range(len(g.controls)):
            controls.append(g.controls[i])

    if not is_controlled:
        var m = Matrix16x16(
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
        )
        for i in range(16):
            var b3 = (i >> 3) & 1
            var b2 = (i >> 2) & 1
            var b1 = (i >> 1) & 1
            var b0 = i & 1
            var b_t: Int
            if target == q3:
                b_t = b3
            elif target == q2:
                b_t = b2
            elif target == q1:
                b_t = b1
            else:
                b_t = b0
            m[i][i] = gate[b_t][b_t]
            var flipped_j: Int
            if target == q3:
                flipped_j = i ^ 8
            elif target == q2:
                flipped_j = i ^ 4
            elif target == q1:
                flipped_j = i ^ 2
            else:
                flipped_j = i ^ 1
            m[flipped_j][i] = gate[1 - b_t][b_t]
        return m

    var m = Matrix16x16(
        r16,
        r16,
        r16,
        r16,
        r16,
        r16,
        r16,
        r16,
        r16,
        r16,
        r16,
        r16,
        r16,
        r16,
        r16,
        r16,
    )
    for i in range(16):
        var b3 = (i >> 3) & 1
        var b2 = (i >> 2) & 1
        var b1 = (i >> 1) & 1
        var b0 = i & 1

        var controls_satisfied = True
        for q_idx in range(num_controls):
            var ctrl = controls[q_idx]
            var val: Int
            if ctrl == q3:
                val = b3
            elif ctrl == q2:
                val = b2
            elif ctrl == q1:
                val = b1
            elif ctrl == q0:
                val = b0
            else:
                controls_satisfied = False
                break
            if val == 0:
                controls_satisfied = False
                break

        if not controls_satisfied:
            m[i][i] = Amplitude(1, 0)
        else:
            var b_t: Int
            if target == q3:
                b_t = b3
            elif target == q2:
                b_t = b2
            elif target == q1:
                b_t = b1
            else:
                b_t = b0

            m[i][i] = gate[b_t][b_t]
            var flipped_j: Int
            if target == q3:
                flipped_j = i ^ 8
            elif target == q2:
                flipped_j = i ^ 4
            elif target == q1:
                flipped_j = i ^ 2
            else:
                flipped_j = i ^ 1
            m[flipped_j][i] = gate[1 - b_t][b_t]
    return m


struct FusedTransformation(Copyable, ImplicitlyCopyable, Movable):
    """A pre-computed fused transformation for efficient execution."""

    var type: Int  # 0: Permutation, 1: Matrix4, 2: Matrix8, 3: Matrix16
    var q0: Int
    var q1: Int
    var q2: Int
    var q3: Int
    var m4: Matrix4x4
    var m8: Matrix8x8
    var m16: Matrix16x16

    fn __init__(out self, is_perm: Bool):
        self.type = 0 if is_perm else -1
        self.q0 = -1
        self.q1 = -1
        self.q2 = -1
        self.q3 = -1
        var r4 = InlineArray[Amplitude, 4](Amplitude(0, 0))
        self.m4 = Matrix4x4(r4, r4, r4, r4)
        var r8 = InlineArray[Amplitude, 8](Amplitude(0, 0))
        self.m8 = Matrix8x8(r8, r8, r8, r8, r8, r8, r8, r8)
        var r16 = InlineArray[Amplitude, 16](Amplitude(0, 0))
        self.m16 = Matrix16x16(
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
        )

    fn __init__(out self, q_high: Int, q_low: Int, mat: Matrix4x4):
        self.type = 1
        self.q0 = q_low
        self.q1 = q_high
        self.q2 = -1
        self.q3 = -1
        self.m4 = mat
        var r8 = InlineArray[Amplitude, 8](Amplitude(0, 0))
        self.m8 = Matrix8x8(r8, r8, r8, r8, r8, r8, r8, r8)
        var r16 = InlineArray[Amplitude, 16](Amplitude(0, 0))
        self.m16 = Matrix16x16(
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
        )

    fn __init__(out self, q_high: Int, q_mid: Int, q_low: Int, mat: Matrix8x8):
        self.type = 2
        self.q0 = q_low
        self.q1 = q_mid
        self.q2 = q_high
        self.q3 = -1
        self.m8 = mat
        var r4 = InlineArray[Amplitude, 4](Amplitude(0, 0))
        self.m4 = Matrix4x4(r4, r4, r4, r4)
        var r16 = InlineArray[Amplitude, 16](Amplitude(0, 0))
        self.m16 = Matrix16x16(
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
            r16,
        )

    fn __init__(out self, q3: Int, q2: Int, q1: Int, q0: Int, mat: Matrix16x16):
        self.type = 3
        self.q0 = q0
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        var r4 = InlineArray[Amplitude, 4](Amplitude(0, 0))
        self.m4 = Matrix4x4(r4, r4, r4, r4)
        var r8 = InlineArray[Amplitude, 8](Amplitude(0, 0))
        self.m8 = Matrix8x8(r8, r8, r8, r8, r8, r8, r8, r8)
        self.m16 = mat

    fn copy(self) -> Self:
        return self


struct FusedV3Group(Copyable, ImplicitlyCopyable, Movable):
    """A pre-analyzed and potentially fused group of transformations for V3 executor.
    """

    var transformations: List[Transformation]
    var is_local: Bool
    var is_radix4: Bool
    var is_radix8: Bool
    var m4: Matrix4x4
    var m8: Matrix8x8
    var q_high: Int
    var q_mid: Int
    var q_low: Int

    fn __init__(
        out self,
        is_local: Bool = False,
        is_radix4: Bool = False,
        is_radix8: Bool = False,
    ):
        self.transformations = List[Transformation]()
        self.is_local = is_local
        self.is_radix4 = is_radix4
        self.is_radix8 = is_radix8
        var r4 = InlineArray[Amplitude, 4](Amplitude(0, 0))
        self.m4 = Matrix4x4(r4, r4, r4, r4)
        var r8 = InlineArray[Amplitude, 8](Amplitude(0, 0))
        self.m8 = Matrix8x8(r8, r8, r8, r8, r8, r8, r8, r8)
        self.q_high = -1
        self.q_mid = -1
        self.q_low = -1

    fn __copyinit__(out self, existing: Self):
        self.transformations = existing.transformations.copy()
        self.is_local = existing.is_local
        self.is_radix4 = existing.is_radix4
        self.is_radix8 = existing.is_radix8
        self.m4 = existing.m4
        self.m8 = existing.m8
        self.q_high = existing.q_high
        self.q_mid = existing.q_mid
        self.q_low = existing.q_low

    fn __moveinit__(out self, deinit existing: Self):
        self.transformations = existing.transformations^
        self.is_local = existing.is_local
        self.is_radix4 = existing.is_radix4
        self.is_radix8 = existing.is_radix8
        self.m4 = existing.m4
        self.m8 = existing.m8
        self.q_high = existing.q_high
        self.q_mid = existing.q_mid
        self.q_low = existing.q_low

    fn copy(self) -> Self:
        return self


struct QuantumRegister(Copyable, Movable):
    """A quantum register representing a named collection of qubits."""

    var name: String
    var start: Int
    var size: Int

    fn __init__(out self, name: String, start: Int, size: Int):
        self.name = name
        self.start = start
        self.size = size

    fn __copyinit__(out self, existing: Self):
        self.name = existing.name
        self.start = existing.start
        self.size = existing.size

    fn __moveinit__(out self, deinit existing: Self):
        self.name = existing.name^
        self.start = existing.start
        self.size = existing.size

    fn __getitem__(self, idx: Int) -> Int:
        """Get the global qubit index for a register qubit."""
        return self.start + idx

    fn qubits(self) -> List[Int]:
        """Return list of all qubit indices in this register."""
        var result = List[Int](capacity=self.size)
        for i in range(self.size):
            result.append(self.start + i)
        return result^


struct QuantumCircuit(Copyable, Movable):
    """A quantum circuit that manages quantum state and transformation operations.

    The circuit maintains a quantum state and a sequence of transformations (gates)
    that can be applied to the state. Supports quantum registers for organizing qubits.
    """

    var transformations: List[Transformation]
    var num_qubits: Int
    var registers: List[QuantumRegister]
    var fused_transformations: List[FusedTransformation]
    var fused_v3_groups: List[FusedV3Group]
    var is_fused: Bool
    var is_transpiled_v3: Bool

    fn __init__(out self, num_qubits: Int):
        """Initialize a quantum circuit with n qubits in the |0⟩ state."""
        self.num_qubits = num_qubits
        self.transformations = List[Transformation]()
        self.registers = List[QuantumRegister]()
        self.fused_transformations = List[FusedTransformation]()
        self.fused_v3_groups = List[FusedV3Group]()
        self.is_fused = False
        self.is_transpiled_v3 = False

    fn __copyinit__(out self, existing: Self):
        self.num_qubits = existing.num_qubits
        self.transformations = List[Transformation](
            capacity=len(existing.transformations)
        )
        for i in range(len(existing.transformations)):
            # Variant copy requires care if types are complex, but generally strict copy works
            # or manual dispatch if needed. Mojo List copy usually works for Variants if variants are copyable.
            self.transformations.append(existing.transformations[i])
        self.registers = List[QuantumRegister](capacity=len(existing.registers))
        for i in range(len(existing.registers)):
            self.registers.append(existing.registers[i].copy())
        self.fused_transformations = List[FusedTransformation](
            capacity=len(existing.fused_transformations)
        )
        for i in range(len(existing.fused_transformations)):
            self.fused_transformations.append(
                existing.fused_transformations[i].copy()
            )
        self.fused_v3_groups = List[FusedV3Group](
            capacity=len(existing.fused_v3_groups)
        )
        for i in range(len(existing.fused_v3_groups)):
            self.fused_v3_groups.append(existing.fused_v3_groups[i].copy())
        self.is_transpiled_v3 = existing.is_transpiled_v3
        self.is_fused = existing.is_fused

    fn __moveinit__(out self, deinit existing: Self):
        self.num_qubits = existing.num_qubits
        self.transformations = existing.transformations^
        self.registers = existing.registers^
        self.fused_transformations = existing.fused_transformations^
        self.fused_v3_groups = existing.fused_v3_groups^
        self.is_fused = existing.is_fused
        self.is_transpiled_v3 = existing.is_transpiled_v3

    fn add(
        mut self,
        gate: Gate,
        target: Int,
        name: String = "unitary",
        arg: FloatType = 0.0,
    ):
        """Add a gate to the circuit on the specified target qubit."""
        self.is_fused = False
        self.transformations.append(GateTransformation(gate, target, name, arg))

    fn add_controlled(
        mut self,
        gate: Gate,
        target: Int,
        control: Int,
        name: String = "unitary",
        arg: FloatType = 0.0,
    ):
        """Add a controlled gate to the circuit."""
        self.transformations.append(
            SingleControlGateTransformation(gate, target, control, name, arg)
        )

    fn add_multi_controlled(
        mut self,
        gate: Gate,
        target: Int,
        var controls: List[Int],
        name: String = "unitary",
        arg: FloatType = 0.0,
    ):
        """Add a multi-controlled gate to the circuit."""
        self.transformations.append(
            MultiControlGateTransformation(gate, target, controls, name, arg)
        )

    fn execute(mut self, mut state: QuantumState):
        execute(state, self)

    fn run(mut self) -> QuantumState:
        """Execute all transformations in the circuit and return the resulting state.

        Returns:
            The quantum state after applying all transformations.
        """
        var state = QuantumState(self.num_qubits)
        execute(state, self)
        return state

    fn _execute_fused(mut self) -> QuantumState:
        """Execute pre-computed fused transformations and return the resulting state.

        Returns:
            The quantum state after applying all fused transformations.
        """
        if not self.is_fused:
            self.fuse()

        var state = QuantumState(self.num_qubits)

        for i in range(len(self.fused_transformations)):
            var ft = self.fused_transformations[i].copy()
            if ft.type == 0:
                bit_reverse_state(state)
            elif ft.type == 1:
                transform_matrix4(state, ft.q1, ft.q0, ft.m4)
            elif ft.type == 2:
                transform_matrix8(state, ft.q2, ft.q1, ft.q0, ft.m8)
            elif ft.type == 3:
                transform_matrix16(state, ft.q3, ft.q2, ft.q1, ft.q0, ft.m16)
        return state

    fn transpile_v3(
        mut self,
        use_radix8: Bool = False,
        fuse_controlled: Bool = False,
        block_log: Int = 11,
    ) raises:
        """Pre-analyze and fuse the circuit using V3 strategy.
        This eliminates the 'Analysis Tax' during the execution loop.
        """
        from butterfly.core.execute_fused_v3 import (
            analyze_and_fuse_v3,
            DEFAULT_BLOCK_LOG,
        )

        var groups = analyze_and_fuse_v3(
            self, use_radix8, fuse_controlled, block_log
        )
        self.fused_v3_groups = groups^
        self.is_transpiled_v3 = True

    fn run_simd_dynamic(mut self) -> QuantumState:
        """
        Execute circuit with SIMD optimizations (runtime dispatch).
        Dispatches to execute_simd[num_qubits] for common sizes.
        """
        var state = QuantumState(self.num_qubits)
        var num_qubits = self.num_qubits

        # Dispatch to SIMD for common qubit counts
        from butterfly.core.execute_simd_dispatch import (
            execute_transformations_simd,
        )

        if num_qubits == 25:
            execute_transformations_simd[1 << 25](state, self.transformations)
        elif num_qubits == 24:
            execute_transformations_simd[1 << 24](state, self.transformations)
        elif num_qubits == 26:
            execute_transformations_simd[1 << 26](state, self.transformations)
        elif num_qubits == 23:
            execute_transformations_simd[1 << 23](state, self.transformations)
        elif num_qubits == 27:
            execute_transformations_simd[1 << 27](state, self.transformations)
        elif num_qubits == 28:
            execute_transformations_simd[1 << 28](state, self.transformations)
        elif num_qubits == 29:
            execute_transformations_simd[1 << 29](state, self.transformations)
        elif num_qubits == 30:
            execute_transformations_simd[1 << 30](state, self.transformations)
        elif num_qubits == 22:
            execute_transformations_simd[1 << 22](state, self.transformations)
        elif num_qubits == 21:
            execute_transformations_simd[1 << 21](state, self.transformations)
        elif num_qubits == 20:
            execute_transformations_simd[1 << 20](state, self.transformations)
        elif num_qubits == 19:
            execute_transformations_simd[1 << 19](state, self.transformations)
        elif num_qubits == 18:
            execute_transformations_simd[1 << 18](state, self.transformations)
        elif num_qubits == 17:
            execute_transformations_simd[1 << 17](state, self.transformations)
        elif num_qubits == 16:
            execute_transformations_simd[1 << 16](state, self.transformations)
        elif num_qubits == 15:
            execute_transformations_simd[1 << 15](state, self.transformations)
        elif num_qubits == 14:
            execute_transformations_simd[1 << 14](state, self.transformations)
        elif num_qubits == 13:
            execute_transformations_simd[1 << 13](state, self.transformations)
        elif num_qubits == 12:
            execute_transformations_simd[1 << 12](state, self.transformations)
        elif num_qubits == 11:
            execute_transformations_simd[1 << 11](state, self.transformations)
        elif num_qubits == 10:
            execute_transformations_simd[1 << 10](state, self.transformations)
        else:
            # Fall back to standard execution for other sizes
            self.execute(state)
        return state

    fn execute_simd_dynamic(mut self, mut state: QuantumState):
        """Execute circuit with SIMD optimizations on provided state (runtime dispatch).
        """
        var num_qubits = self.num_qubits
        if num_qubits == 25:
            execute_transformations_simd[1 << 25](state, self.transformations)
        elif num_qubits == 24:
            execute_transformations_simd[1 << 24](state, self.transformations)
        elif num_qubits == 26:
            execute_transformations_simd[1 << 26](state, self.transformations)
        elif num_qubits == 22:
            execute_transformations_simd[1 << 22](state, self.transformations)
        elif num_qubits == 20:
            execute_transformations_simd[1 << 20](state, self.transformations)
        elif num_qubits == 15:
            execute_transformations_simd[1 << 15](state, self.transformations)
        elif num_qubits == 10:
            execute_transformations_simd[1 << 10](state, self.transformations)
        else:
            self.execute(state)

    fn run_simd_v2_dynamic(mut self) -> QuantumState:
        """
        Execute circuit with SIMD v2 optimizations (runtime dispatch).
        Optimized indexing and chunked kernels.
        """
        var state = QuantumState(self.num_qubits)
        var num_qubits = self.num_qubits

        # Dispatch to SIMD v2 for common qubit counts
        from butterfly.core.execute_simd_v2_dispatch import (
            execute_transformations_simd_v2,
        )

        if num_qubits == 25:
            execute_transformations_simd_v2[1 << 25](
                state, self.transformations
            )
        elif num_qubits == 24:
            execute_transformations_simd_v2[1 << 24](
                state, self.transformations
            )
        elif num_qubits == 26:
            execute_transformations_simd_v2[1 << 26](
                state, self.transformations
            )
        elif num_qubits == 23:
            execute_transformations_simd_v2[1 << 23](
                state, self.transformations
            )
        elif num_qubits == 27:
            execute_transformations_simd_v2[1 << 27](
                state, self.transformations
            )
        elif num_qubits == 28:
            execute_transformations_simd_v2[1 << 28](
                state, self.transformations
            )
        elif num_qubits == 29:
            execute_transformations_simd_v2[1 << 29](
                state, self.transformations
            )
        elif num_qubits == 30:
            execute_transformations_simd_v2[1 << 30](
                state, self.transformations
            )
        elif num_qubits == 22:
            execute_transformations_simd_v2[1 << 22](
                state, self.transformations
            )
        elif num_qubits == 21:
            execute_transformations_simd_v2[1 << 21](
                state, self.transformations
            )
        elif num_qubits == 20:
            execute_transformations_simd_v2[1 << 20](
                state, self.transformations
            )
        elif num_qubits == 19:
            execute_transformations_simd_v2[1 << 19](
                state, self.transformations
            )
        elif num_qubits == 18:
            execute_transformations_simd_v2[1 << 18](
                state, self.transformations
            )
        elif num_qubits == 17:
            execute_transformations_simd_v2[1 << 17](
                state, self.transformations
            )
        elif num_qubits == 16:
            execute_transformations_simd_v2[1 << 16](
                state, self.transformations
            )
        elif num_qubits == 15:
            execute_transformations_simd_v2[1 << 15](
                state, self.transformations
            )
        elif num_qubits == 14:
            execute_transformations_simd_v2[1 << 14](
                state, self.transformations
            )
        elif num_qubits == 13:
            execute_transformations_simd_v2[1 << 13](
                state, self.transformations
            )
        elif num_qubits == 12:
            execute_transformations_simd_v2[1 << 12](
                state, self.transformations
            )
        elif num_qubits == 11:
            execute_transformations_simd_v2[1 << 11](
                state, self.transformations
            )
        elif num_qubits == 10:
            execute_transformations_simd_v2[1 << 10](
                state, self.transformations
            )
        else:
            # Fall back to execute_simd() if v2 not available or requested
            state = self.run_simd_dynamic()
        return state

    fn execute_simd_v2_dynamic(mut self, mut state: QuantumState):
        """Execute circuit with SIMD v2 optimizations on provided state (runtime dispatch).
        """
        var num_qubits = self.num_qubits

        # Dispatch to SIMD v2 for common qubit counts
        from butterfly.core.execute_simd_v2_dispatch import (
            execute_transformations_simd_v2,
        )

        if num_qubits == 25:
            execute_transformations_simd_v2[1 << 25](
                state, self.transformations
            )
        elif num_qubits == 24:
            execute_transformations_simd_v2[1 << 24](
                state, self.transformations
            )
        elif num_qubits == 26:
            execute_transformations_simd_v2[1 << 26](
                state, self.transformations
            )
        elif num_qubits == 23:
            execute_transformations_simd_v2[1 << 23](
                state, self.transformations
            )
        elif num_qubits == 27:
            execute_transformations_simd_v2[1 << 27](
                state, self.transformations
            )
        elif num_qubits == 28:
            execute_transformations_simd_v2[1 << 28](
                state, self.transformations
            )
        elif num_qubits == 29:
            execute_transformations_simd_v2[1 << 29](
                state, self.transformations
            )
        elif num_qubits == 30:
            execute_transformations_simd_v2[1 << 30](
                state, self.transformations
            )
        elif num_qubits == 22:
            execute_transformations_simd_v2[1 << 22](
                state, self.transformations
            )
        elif num_qubits == 21:
            execute_transformations_simd_v2[1 << 21](
                state, self.transformations
            )
        elif num_qubits == 20:
            execute_transformations_simd_v2[1 << 20](
                state, self.transformations
            )
        elif num_qubits == 19:
            execute_transformations_simd_v2[1 << 19](
                state, self.transformations
            )
        elif num_qubits == 18:
            execute_transformations_simd_v2[1 << 18](
                state, self.transformations
            )
        elif num_qubits == 17:
            execute_transformations_simd_v2[1 << 17](
                state, self.transformations
            )
        elif num_qubits == 16:
            execute_transformations_simd_v2[1 << 16](
                state, self.transformations
            )
        elif num_qubits == 15:
            execute_transformations_simd_v2[1 << 15](
                state, self.transformations
            )
        elif num_qubits == 14:
            execute_transformations_simd_v2[1 << 14](
                state, self.transformations
            )
        elif num_qubits == 13:
            execute_transformations_simd_v2[1 << 13](
                state, self.transformations
            )
        elif num_qubits == 12:
            execute_transformations_simd_v2[1 << 12](
                state, self.transformations
            )
        elif num_qubits == 11:
            execute_transformations_simd_v2[1 << 11](
                state, self.transformations
            )
        elif num_qubits == 10:
            execute_transformations_simd_v2[1 << 10](
                state, self.transformations
            )
        else:
            # Fall back to generic execution
            self.execute(state)

    fn apply_transformation_super_fast(
        self, mut state: QuantumState, t: Transformation
    ):
        """
        Apply transformation with all optimizations enabled.
        Uses specialized SIMD kernels for common gates.
        """
        if t.isa[BitReversalTransformation]():
            bit_reverse_state(state)
            return

        if t.isa[SingleControlGateTransformation]():
            var g = t[SingleControlGateTransformation].copy()
            # Use optimized controlled-H if applicable
            if is_h(g.gate):
                from butterfly.core.c_transform_fast import c_transform_h_simd

                c_transform_h_simd(state, g.control, g.target)
            else:
                c_transform(state, g.control, g.target, g.gate)
        elif t.isa[MultiControlGateTransformation]():
            var g = t[MultiControlGateTransformation].copy()
            mc_transform_interval(state, g.controls, g.target, g.gate)
        elif t.isa[GateTransformation]():
            var g = t[GateTransformation].copy()
            if is_h(g.gate):
                from butterfly.core.state import transform_h_block_style

                transform_h_block_style(state, g.target)
            else:
                transform(state, g.target, g.gate)

    fn run_simd_unfused_dynamic(mut self) -> QuantumState:
        """
        Execute with SIMD optimizations but without fusion (runtime dispatch).
        Useful for comparing optimization impact vs fusion impact.
        """
        var state = QuantumState(self.num_qubits)
        for i in range(len(self.transformations)):
            var t = self.transformations[i].copy()
            self.apply_transformation_super_fast(state, t)
        return state

    fn execute_simd_unfused_dynamic(mut self, mut state: QuantumState):
        """Execute with SIMD optimizations but without fusion on provided state (runtime dispatch).
        """
        # For v0.1, fallback to generic for dynamic unfused execution
        self.execute(state)

    fn fuse(mut self):
        """Perform greedy fusion and pre-compute transformation matrices."""
        if self.is_fused:
            return

        self.fused_transformations = List[FusedTransformation]()
        var i = 0
        var n_ops = len(self.transformations)

        while i < n_ops:
            var t = self.transformations[i].copy()

            if is_permutation(t):
                self.fused_transformations.append(
                    FusedTransformation(is_perm=True)
                )
                i += 1
                continue

            var q_set = get_involved_qubits(t)
            var j = i + 1
            while j < n_ops:
                var nt = self.transformations[j].copy()
                if is_permutation(nt):
                    break

                var nq = get_involved_qubits(nt)
                var combined = q_set.copy()
                for q in nq:
                    var found = False
                    for cq in q_set:
                        if q == cq:
                            found = True
                            break
                    if not found:
                        combined.append(q)

                if len(combined) <= 3:  # Default fusion limit = 3
                    q_set = combined^
                    j += 1
                else:
                    break

            if j == i + 1:
                # Still use matrix4 for single non-controlled gate if not already fused
                # but for simplicity we can just use transform_matrix4 or similar
                # Let's use get_as_matrix4x4 with a dummy high qubit if needed
                var q_high: Int
                var q_low: Int
                if len(q_set) == 2:
                    q_high = max(q_set[0], q_set[1])
                    q_low = min(q_set[0], q_set[1])
                else:
                    q_low = q_set[0]
                    q_high = q_low + 1
                    if q_high >= self.num_qubits:
                        q_high = q_low - 1
                    if q_high < 0:
                        q_high = 0
                    var real_high = max(q_high, q_low)
                    var real_low = min(q_high, q_low)
                    q_high = real_high
                    q_low = real_low

                var mat = get_as_matrix4x4(t, q_high, q_low)
                self.fused_transformations.append(
                    FusedTransformation(q_high, q_low, mat)
                )
                i += 1
                continue

            if len(q_set) == 4:
                # Sort q_set descending
                for ii in range(3):
                    for jj in range(3 - ii):
                        if q_set[jj] < q_set[jj + 1]:
                            var tmp = q_set[jj]
                            q_set[jj] = q_set[jj + 1]
                            q_set[jj + 1] = tmp
                var q3 = q_set[0]
                var q2 = q_set[1]
                var q1 = q_set[2]
                var q0 = q_set[3]

                var fused_mat = get_as_matrix16x16(
                    self.transformations[i].copy(), q3, q2, q1, q0
                )
                for k in range(i + 1, j):
                    var next_mat = get_as_matrix16x16(
                        self.transformations[k].copy(), q3, q2, q1, q0
                    )
                    fused_mat = matmul_matrix16x16(next_mat, fused_mat)

                self.fused_transformations.append(
                    FusedTransformation(q3, q2, q1, q0, fused_mat)
                )
                i = j
            elif len(q_set) == 3:
                var q_high = max(q_set[0], max(q_set[1], q_set[2]))
                var q_low = min(q_set[0], min(q_set[1], q_set[2]))
                var q_mid = q_set[0] + q_set[1] + q_set[2] - q_high - q_low

                var fused_mat = get_as_matrix8x8(
                    self.transformations[i].copy(), q_high, q_mid, q_low
                )
                for k in range(i + 1, j):
                    var next_mat = get_as_matrix8x8(
                        self.transformations[k].copy(), q_high, q_mid, q_low
                    )
                    fused_mat = matmul_matrix8x8(next_mat, fused_mat)

                self.fused_transformations.append(
                    FusedTransformation(q_high, q_mid, q_low, fused_mat)
                )
                i = j
            else:
                # 2 qubits
                var q_high = max(q_set[0], q_set[1])
                var q_low = min(q_set[0], q_set[1])

                var fused_mat = get_as_matrix4x4(
                    self.transformations[i].copy(), q_high, q_low
                )
                for k in range(i + 1, j):
                    var next_mat = get_as_matrix4x4(
                        self.transformations[k].copy(), q_high, q_low
                    )
                    fused_mat = matmul_matrix4x4(next_mat, fused_mat)

                self.fused_transformations.append(
                    FusedTransformation(q_high, q_low, fused_mat)
                )
                i = j

        self.is_fused = True

    fn run_fused_v3_dynamic(mut self) -> QuantumState:
        """Execute transformations with fused_v3 optimization (runtime dispatch).
        """
        if not self.is_fused:
            self.fuse()
        return self._execute_fused()

    fn execute_fused_v3_dynamic(mut self, mut state: QuantumState):
        """Execute transformations with fused_v3 optimization on provided state (runtime dispatch).
        """
        var num_qubits = self.num_qubits

        # Dispatch to fused_v3 for common qubit counts
        from butterfly.core.execute_fused_v3 import execute_fused_v3

        if num_qubits == 25:
            execute_fused_v3[1 << 25](state, self)
        elif num_qubits == 24:
            execute_fused_v3[1 << 24](state, self)
        elif num_qubits == 26:
            execute_fused_v3[1 << 26](state, self)
        elif num_qubits == 23:
            execute_fused_v3[1 << 23](state, self)
        elif num_qubits == 27:
            execute_fused_v3[1 << 27](state, self)
        elif num_qubits == 28:
            execute_fused_v3[1 << 28](state, self)
        elif num_qubits == 29:
            execute_fused_v3[1 << 29](state, self)
        elif num_qubits == 30:
            execute_fused_v3[1 << 30](state, self)
        elif num_qubits == 22:
            execute_fused_v3[1 << 22](state, self)
        elif num_qubits == 21:
            execute_fused_v3[1 << 21](state, self)
        elif num_qubits == 20:
            execute_fused_v3[1 << 20](state, self)
        elif num_qubits == 19:
            execute_fused_v3[1 << 19](state, self)
        elif num_qubits == 18:
            execute_fused_v3[1 << 18](state, self)
        elif num_qubits == 17:
            execute_fused_v3[1 << 17](state, self)
        elif num_qubits == 16:
            execute_fused_v3[1 << 16](state, self)
        elif num_qubits == 15:
            execute_fused_v3[1 << 15](state, self)
        elif num_qubits == 14:
            execute_fused_v3[1 << 14](state, self)
        elif num_qubits == 13:
            execute_fused_v3[1 << 13](state, self)
        elif num_qubits == 12:
            execute_fused_v3[1 << 12](state, self)
        elif num_qubits == 11:
            execute_fused_v3[1 << 11](state, self)
        elif num_qubits == 10:
            execute_fused_v3[1 << 10](state, self)
        else:
            # Fall back to generic execution
            self.execute(state)

    fn run_with_strategy(mut self, strategy: ExecutionStrategy) -> QuantumState:
        """Execute circuit with specified strategy and return new state.

        Args:
            strategy: Execution strategy to use.

        Returns:
            The quantum state after applying all transformations.
        """
        if strategy.isa[Generic]():
            return self.run()
        elif strategy.isa[SIMD]():
            return self.run_simd_dynamic()
        elif strategy.isa[SIMDv2]():
            return self.run_simd_v2_dynamic()
        elif strategy.isa[FusedV3]():
            return self.run_fused_v3_dynamic()
        else:
            return self.run()  # fallback to generic

    fn execute_with_strategy[
        n: Int, strategy: ExecutionStrategy
    ](mut self, mut state: QuantumState):
        """Execute circuit with specified strategy (fully compile-time specialized).

        Args:
            state: The quantum state to transform (modified in place).

        Parameters:
            n: Number of qubits (compile-time constant).
            strategy: Execution strategy to use (compile-time constant).

        Note:
            Zero-overhead abstraction - both n and strategy known at compile time.
            For runtime strategy selection, use the overload with strategy as an argument.
        """
        execute_with_strategy[n, strategy](state, self)

    fn execute_with_strategy[
        n: Int
    ](mut self, mut state: QuantumState, strategy: ExecutionStrategy):
        """Execute circuit with specified strategy (compile-time specialized).

        Args:
            state: The quantum state to transform (modified in place).
            strategy: Execution strategy to use.

        Parameters:
            n: Number of qubits (compile-time constant).

        Note:
            This is the compile-time specialized version. For dynamic dispatch,
            use execute_with_strategy_dynamic().
        """
        execute_with_strategy[n](state, self, strategy)

    fn execute_with_strategy_dynamic(
        mut self, mut state: QuantumState, strategy: ExecutionStrategy
    ):
        """Execute circuit with specified strategy on provided state (dynamic dispatch).

        Args:
            state: The quantum state to transform (modified in place).
            strategy: Execution strategy to use.
        """
        if strategy.isa[Generic]():
            self.execute(state)
        elif strategy.isa[SIMD]():
            self.execute_simd_dynamic(state)
        elif strategy.isa[SIMDv2]():
            self.execute_simd_v2_dynamic(state)
        elif strategy.isa[FusedV3]():
            self.execute_fused_v3_dynamic(state)
        else:
            self.execute(state)  # fallback to generic

    fn run_adaptive(mut self) -> QuantumState:
        """Execute circuit with automatically selected optimal strategy.

        Selects the best executor based on circuit size:
        - n ≤ 9: execute_simd (fastest for small circuits)
        - n = 10-12: execute (generic, fastest for medium circuits)
        - n ≥ 13: execute_simd_v2 (fastest for large circuits)

        Returns:
            The resulting quantum state after execution.
        """
        from butterfly.core.adaptive_strategy import get_optimal_strategy

        var state = QuantumState(self.num_qubits)
        var strategy = get_optimal_strategy(self.num_qubits)
        self.execute_with_strategy_dynamic(state, strategy)
        return state^

    fn clear_transformations(mut self):
        """Clear all transformations from the circuit."""
        self.transformations = List[Transformation]()

    fn num_transformations(self) -> Int:
        """Return the number of transformations in the circuit."""
        return len(self.transformations)

    fn inverse(self) -> QuantumCircuit:
        """Return a new QuantumCircuit that is the inverse of this circuit."""
        var res = QuantumCircuit(self.num_qubits)
        # Copy registers
        for i in range(len(self.registers)):
            var r = self.registers[i].copy()
            _ = res.add_register(r.name, r.size)

        # Iterate transformations in reverse
        for i in range(len(self.transformations) - 1, -1, -1):
            var t = self.transformations[i].copy()
            if t.isa[GateTransformation]():
                var g = t[GateTransformation].copy()
                if g.name == "h":
                    res.h(g.target)
                elif g.name == "x":
                    res.x(g.target)
                elif g.name == "y":
                    res.y(g.target)
                elif g.name == "z":
                    res.z(g.target)
                elif g.name == "rx":
                    res.rx(g.target, -g.arg)
                elif g.name == "ry":
                    res.ry(g.target, -g.arg)
                elif g.name == "rz":
                    res.rz(g.target, -g.arg)
                elif g.name == "p":
                    res.p(g.target, -g.arg)
                else:
                    # Arbitrary unitary
                    res.unitary(dagger(g.gate), g.target)
            elif t.isa[SingleControlGateTransformation]():
                var g = t[SingleControlGateTransformation].copy()
                if g.name == "h":
                    res.ch(g.target, g.control)
                elif g.name == "x":
                    res.cx(g.target, g.control)
                elif g.name == "y":
                    res.cy(g.target, g.control)
                elif g.name == "z":
                    res.cz(g.target, g.control)
                elif g.name == "rx":
                    res.crx(g.target, g.control, -g.arg)
                elif g.name == "ry":
                    res.cry(g.target, g.control, -g.arg)
                elif g.name == "rz":
                    res.crz(g.target, g.control, -g.arg)
                elif g.name == "p":
                    res.cp(g.control, g.target, -g.arg)
                else:
                    res.c_unitary(dagger(g.gate), g.control, g.target)
            elif t.isa[MultiControlGateTransformation]():
                var g = t[MultiControlGateTransformation].copy()
                if g.name == "x":
                    res.mcx(g.controls.copy(), g.target)
                else:
                    res.add_multi_controlled(
                        dagger(g.gate),
                        g.target,
                        g.controls.copy(),
                        g.name,
                        -g.arg,
                    )
            elif t.isa[UnitaryTransformation]():
                var g = t[UnitaryTransformation].copy()
                if g.m == 1:
                    res.unitary(dagger(g.u, g.m), g.target, g.name)
                else:
                    # Registry-based
                    res.append_u(
                        dagger(g.u, g.m),
                        QuantumRegister("", g.target, g.m),
                        g.name,
                    )
            elif t.isa[ControlledUnitaryTransformation]():
                var g = t[ControlledUnitaryTransformation].copy()
                if g.m == 1:
                    res.c_unitary(dagger(g.u, g.m), g.control, g.target, g.name)
                else:
                    res.c_append_u(
                        dagger(g.u, g.m),
                        g.control,
                        QuantumRegister("", g.target, g.m),
                        g.name,
                    )
            elif t.isa[BitReversalTransformation]():
                res.bit_reverse()

        return res^

    # Gate-specific methods
    fn h(mut self, target: Int):
        """Apply Hadamard gate to target qubit."""
        self.add(H, target, "h")

    fn x(mut self, target: Int):
        """Apply Pauli-X gate to target qubit."""
        self.add(X, target, "x")

    fn y(mut self, target: Int):
        """Apply Pauli-Y gate to target qubit."""
        self.add(Y, target, "y")

    fn z(mut self, target: Int):
        """Apply Pauli-Z gate to target qubit."""
        self.add(Z, target, "z")

    fn rx(mut self, target: Int, theta: FloatType):
        """Apply RX rotation gate to target qubit."""
        self.add(RX(theta), target, "rx", theta)

    fn ry(mut self, target: Int, theta: FloatType):
        """Apply RY rotation gate to target qubit."""
        self.add(RY(theta), target, "ry", theta)

    fn rz(mut self, target: Int, theta: FloatType):
        """Apply RZ rotation gate to target qubit."""
        self.add(RZ(theta), target, "rz", theta)

    fn p(mut self, target: Int, theta: FloatType):
        """Apply phase gate to target qubit."""
        self.add(P(theta), target, "p", theta)

    # Controlled versions
    fn ch(mut self, target: Int, control: Int):
        """Apply controlled Hadamard gate."""
        self.add_controlled(H, target, control, "h")

    fn cx(mut self, target: Int, control: Int):
        """Apply controlled X (CNOT) gate."""
        self.add_controlled(X, target, control, "x")

    fn cy(mut self, target: Int, control: Int):
        """Apply controlled Y gate."""
        self.add_controlled(Y, target, control, "y")

    fn cz(mut self, target: Int, control: Int):
        """Apply controlled Z gate."""
        self.add_controlled(Z, target, control, "z")

    fn crx(mut self, target: Int, control: Int, theta: FloatType):
        """Apply controlled RX rotation gate."""
        self.add_controlled(RX(theta), target, control, "rx", theta)

    fn cry(mut self, target: Int, control: Int, theta: FloatType):
        """Apply controlled RY rotation gate."""
        self.add_controlled(RY(theta), target, control, "ry", theta)

    fn crz(mut self, target: Int, control: Int, theta: FloatType):
        """Apply controlled RZ rotation gate."""
        self.add_controlled(RZ(theta), target, control, "rz", theta)

    fn cp(mut self, control: Int, target: Int, theta: FloatType):
        """Apply controlled phase gate."""
        self.add_controlled(P(theta), target, control, "p", theta)

    fn cft(
        mut self,
        mut state: QuantumState,
        targets: List[Int],
        inverse: Bool = False,
        do_swap: Bool = True,
    ):
        """
        Classical Fourier Transform (CFT) method.
        Directly applies FFT butterflies to the amplitudes of the target qubits.

        Args:
            state: The QuantumState to transform.
            targets: The list of target qubits.
            inverse: If True, applies the inverse transform.
            do_swap: If True, applies bit-reversal swapping to the target qubits.
        """
        var k = len(targets)
        if k == 0:
            return

        # Sort targets ascending for consistent subspace mapping
        var sorted_targets = targets.copy()
        for i in range(k):
            for j in range(i + 1, k):
                if sorted_targets[i] > sorted_targets[j]:
                    var temp = sorted_targets[i]
                    sorted_targets[i] = sorted_targets[j]
                    sorted_targets[j] = temp

        # Generate twiddle factors for size 2^k
        from butterfly.core.classical_fft import generate_factors

        # FIXME: Re-enable CFT classical shortcut once aliasing lints are resolved.
        # var factors_pair = generate_factors(1 << k, inverse)
        # var ptr_fac_re = factors_pair[0].unsafe_ptr()
        # var ptr_fac_im = factors_pair[1].unsafe_ptr()

        # # DIF FFT: Process from highest qubit to lowest qubit
        # for i in reversed(range(k)):
        #     var target = sorted_targets[i]
        #     # Subspace bits are all targets lower than 'target'
        #     var subspace_indices = List[Int]()
        #     for s_idx in range(i):
        #         subspace_indices.append(sorted_targets[s_idx])

        #     # Twiddle stride in the factors table: 1, 2, 4, ..., 2^(k-1)
        #     var tw_stride = 1 << (k - i - 1)

        #     apply_cft_stage(
        #         state,
        #         target,
        #         subspace_indices,
        #         tw_stride,
        #         ptr_fac_re,
        #         ptr_fac_im,
        #         inverse,
        #     )

        # if do_swap:
        #     partial_bit_reverse_state(state, targets)

        # # Normalization
        # var scale = 1.0 / sqrt(Float64(1 << k))
        # for i in range(state.size()):
        #     state.re[i] *= scale
        #     state.im[i] *= scale

    fn mcx(mut self, var controls: List[Int], target: Int):
        """Apply multi-controlled X gate."""
        self.add_multi_controlled(X, target, controls^, "x")

    fn mcp(mut self, theta: FloatType, var controls: List[Int], target: Int):
        """Apply multi-controlled phase gate."""
        self.add_multi_controlled(P(theta), target, controls^, "p", theta)

    fn swap(mut self, q1: Int, q2: Int):
        """Apply SWAP gate between two qubits using 3 CNOTs."""
        self.cx(q2, q1)
        self.cx(q1, q2)
        self.cx(q2, q1)

    fn mswap(mut self, targets: List[Int]):
        """Apply multiple swaps to reverse the order of qubits in the target list.
        """
        var n = len(targets)
        for j in range(n // 2):
            self.swap(targets[j], targets[n - 1 - j])

    fn mswap(mut self, register: QuantumRegister):
        """Apply multiple swaps to reverse the order of qubits in a register."""
        var n = register.size
        for j in range(n // 2):
            self.swap(register.start + j, register.start + n - 1 - j)

    fn qft(
        mut self,
        register: QuantumRegister,
        reversed: Bool = False,
        swap: Bool = True,
    ):
        """Apply Quantum Fourier Transform to a register."""
        var targets = List[Int]()
        if reversed:
            for i in range(register.size):
                targets.append(register.start + register.size - 1 - i)
        else:
            for i in range(register.size):
                targets.append(register.start + i)
        _qft(self, targets, swap)

    fn iqft(
        mut self,
        register: QuantumRegister,
        reversed: Bool = False,
        swap: Bool = True,
    ):
        """Apply Inverse Quantum Fourier Transform to a register."""
        var targets = List[Int]()
        if reversed:
            for i in range(register.size):
                targets.append(register.start + register.size - 1 - i)
        else:
            for i in range(register.size):
                targets.append(register.start + i)

        _iqft(self, targets, swap)

    fn bit_reverse(mut self):
        """Add an efficient bit-reversal operation to the circuit."""
        self.is_fused = False
        self.transformations.append(BitReversalTransformation())

    fn diagonal_phase_flip(
        mut self, items: List[Int], offset: Int = 0, size: Int = 0
    ):
        """Add a diagonal transformation that flips the phase of specified indices.
        """
        self.is_fused = False
        self.transformations.append(
            DiagonalTransformation(items.copy(), offset, size)
        )

    # Register management
    fn add_register(mut self, name: String, size: Int) -> QuantumRegister:
        """Add a new quantum register to the circuit.

        Returns the created register which can be used to reference qubits.
        """
        var start = 0
        for i in range(len(self.registers)):
            var reg = self.registers[i].copy()
            start = max(start, reg.start + reg.size)

        var reg = QuantumRegister(name, start, size)
        self.registers.append(reg^)
        return QuantumRegister(name, start, size)

    fn get_register(self, name: String) raises -> QuantumRegister:
        """Get a register by name."""
        for i in range(len(self.registers)):
            if self.registers[i].name == name:
                return self.registers[i].copy()
        raise Error("Register not found: " + name)

    fn num_registers(self) -> Int:
        """Return the number of registers in the circuit."""
        return len(self.registers)

    fn unitary(
        mut self, var u: List[Amplitude], target: Int, name: String = "unitary"
    ):
        """Add an arbitrary unitary transformation acting on a single qubit."""
        self.is_fused = False
        self.transformations.append(UnitaryTransformation(u^, target, 1, name))

    fn unitary(mut self, gate: Gate, target: Int, name: String = "unitary"):
        """Add a single-qubit gate to the circuit via a unitary matrix (Gate).
        """
        var u = List[Amplitude](capacity=4)
        u.append(gate[0][0])
        u.append(gate[0][1])
        u.append(gate[1][0])
        u.append(gate[1][1])
        self.unitary(u^, target, name)

    fn u(mut self, var u: List[Amplitude], target: Int):
        """Add an arbitrary unitary transformation acting on a single qubit (shorthand).
        """
        self.unitary(u^, target)

    fn append_u(
        mut self,
        var u: List[Amplitude],
        register: QuantumRegister,
        name: String = "unitary",
    ):
        """Add an arbitrary unitary transformation acting on an entire register.
        """
        if len(u) != (1 << (2 * register.size)):
            print(
                "Warning: Unitary size mismatch. Expected",
                (1 << (2 * register.size)),
                "got",
                len(u),
            )
        self.is_fused = False
        self.transformations.append(
            UnitaryTransformation(u^, register.start, register.size, name)
        )

    fn c_unitary(
        mut self,
        var u: List[Amplitude],
        control: Int,
        target: Int,
        name: String = "unitary",
    ):
        """Add an arbitrary controlled unitary transformation acting on a single target.
        """
        self.is_fused = False
        self.transformations.append(
            ControlledUnitaryTransformation(u^, target, control, 1, name)
        )

    fn c_unitary(
        mut self,
        gate: Gate,
        control: Int,
        target: Int,
        name: String = "unitary",
    ):
        """Add a single-qubit controlled gate via a unitary matrix (Gate)."""
        var u = List[Amplitude](capacity=4)
        u.append(gate[0][0])
        u.append(gate[0][1])
        u.append(gate[1][0])
        u.append(gate[1][1])
        self.c_unitary(u^, control, target, name)

    fn cu(mut self, var u: List[Amplitude], control: Int, target: Int):
        """Add an arbitrary controlled unitary acting on a single target (shorthand).
        """
        self.c_unitary(u^, control, target)

    fn c_append_u(
        mut self,
        var u: List[Amplitude],
        control: Int,
        register: QuantumRegister,
        name: String = "unitary",
    ):
        """Add an arbitrary controlled unitary transformation acting on a register.
        """
        if len(u) != (1 << (2 * register.size)):
            print(
                "Warning: Unitary size mismatch. Expected",
                (1 << (2 * register.size)),
                "got",
                len(u),
            )
        self.is_fused = False
        self.transformations.append(
            ControlledUnitaryTransformation(
                u^, register.start, control, register.size, name
            )
        )

    fn append_circuit(mut self, other: QuantumCircuit):
        """Append a circuit to the current one.

        Assumes the qubits map one-to-one (offset 0).
        """
        self.append_circuit(other, QuantumRegister("", 0, other.num_qubits))

    fn append_circuit(
        mut self, other: QuantumCircuit, target_register: QuantumRegister
    ):
        """Append a circuit to the current one, mapping its qubits to the target register.
        """
        var offset = target_register.start
        self.is_fused = False
        for i in range(len(other.transformations)):
            var t = other.transformations[i].copy()
            if t.isa[GateTransformation]():
                var g = t[GateTransformation].copy()
                self.add(g.gate, g.target + offset)
            elif t.isa[SingleControlGateTransformation]():
                var g = t[SingleControlGateTransformation].copy()
                self.add_controlled(
                    g.gate, g.target + offset, g.control + offset
                )
            elif t.isa[MultiControlGateTransformation]():
                var g = t[MultiControlGateTransformation].copy()
                var mapped_controls = List[Int](capacity=len(g.controls))
                for j in range(len(g.controls)):
                    mapped_controls.append(g.controls[j] + offset)
                self.add_multi_controlled(
                    g.gate, g.target + offset, mapped_controls^
                )
            elif t.isa[UnitaryTransformation]():
                var g = t[UnitaryTransformation].copy()
                self.transformations.append(
                    UnitaryTransformation(g.u.copy(), g.target + offset, g.m)
                )
            elif t.isa[DiagonalTransformation]():
                var g = t[DiagonalTransformation].copy()
                self.diagonal_phase_flip(
                    g.items.copy(), g.offset + offset, g.size
                )
            elif t.isa[ControlledUnitaryTransformation]():
                var g = t[ControlledUnitaryTransformation].copy()
                self.transformations.append(
                    ControlledUnitaryTransformation(
                        g.u.copy(), g.target + offset, g.control + offset, g.m
                    )
                )
            elif t.isa[BitReversalTransformation]():
                self.bit_reverse()

    fn c_append_circuit(
        mut self,
        other: QuantumCircuit,
        target_register: QuantumRegister,
        control: Int,
    ):
        """Append a circuit, controlled by a single qubit."""
        var controls = List[Int]()
        controls.append(control)
        self.mc_append_circuit(other, target_register, controls^)

    fn mc_append_circuit(
        mut self,
        other: QuantumCircuit,
        target_register: QuantumRegister,
        var controls: List[Int],
    ):
        """Append a circuit, controlled by multiple qubits."""
        var offset = target_register.start
        self.is_fused = False
        for i in range(len(other.transformations)):
            var t = other.transformations[i].copy()
            if t.isa[GateTransformation]():
                var g = t[GateTransformation].copy()
                var new_controls = List[Int](capacity=len(controls))
                for j in range(len(controls)):
                    new_controls.append(controls[j])
                self.add_multi_controlled(
                    g.gate, g.target + offset, new_controls^
                )
            elif t.isa[SingleControlGateTransformation]():
                var g = t[SingleControlGateTransformation].copy()
                var new_controls = List[Int](capacity=len(controls) + 1)
                for j in range(len(controls)):
                    new_controls.append(controls[j])
                new_controls.append(g.control + offset)
                self.add_multi_controlled(
                    g.gate, g.target + offset, new_controls^
                )
            elif t.isa[MultiControlGateTransformation]():
                var g = t[MultiControlGateTransformation].copy()
                var new_controls = List[Int](
                    capacity=len(controls) + len(g.controls)
                )
                for j in range(len(controls)):
                    new_controls.append(controls[j])
                for j in range(len(g.controls)):
                    new_controls.append(g.controls[j] + offset)
                self.add_multi_controlled(
                    g.gate, g.target + offset, new_controls^
                )
            elif t.isa[UnitaryTransformation]():
                var g = t[UnitaryTransformation].copy()
                var new_controls = List[Int](capacity=len(controls))
                for j in range(len(controls)):
                    new_controls.append(controls[j])
                self.add_multi_controlled_unitary(
                    g.u.copy(), g.target + offset, new_controls^, g.m
                )
            elif t.isa[ControlledUnitaryTransformation]():
                var g = t[ControlledUnitaryTransformation].copy()
                var new_controls = List[Int](capacity=len(controls) + 1)
                for j in range(len(controls)):
                    new_controls.append(controls[j])
                new_controls.append(g.control + offset)
                self.add_multi_controlled_unitary(
                    g.u.copy(), g.target + offset, new_controls^, g.m
                )
            elif t.isa[BitReversalTransformation]():
                # Bit reversal doesn't easily support control, skip for now
                pass

    # Multi-controlled Unitary helper
    fn add_multi_controlled_unitary(
        mut self,
        var u: List[Amplitude],
        target: Int,
        var controls: List[Int],
        m: Int,
    ):
        """Add a multi-controlled unitary."""
        if len(controls) == 1:
            self.transformations.append(
                ControlledUnitaryTransformation(u^, target, controls[0], m)
            )
        elif len(controls) == 0:
            self.transformations.append(UnitaryTransformation(u^, target, m))
        else:
            # Placeholder: Multi-controlled arbitrary unitary not yet fully supported
            # for qubit counts > 1. For now, we only support single control.
            self.transformations.append(
                ControlledUnitaryTransformation(u^, target, controls[0], m)
            )

    fn execute_fused_v3[n: Int](self, mut state: QuantumState):
        """Execute a circuit on a quantum state using fused transforms.

        Args:
            state: The quantum state to transform (modified in place).
        """
        # Validate that circuit qubits match state size
        var expected_size = 1 << self.num_qubits
        if state.size() != expected_size:
            print(
                "Error: Circuit has",
                self.num_qubits,
                "qubits (expects state size",
                expected_size,
                ") but state has size",
                state.size(),
            )
            return

        # Use fused executor
        execute_fused_v3[1 << n](state, self)


alias Circuit = QuantumCircuit
alias Register = QuantumRegister


# Internal QFT/IQFT helper functions (use QuantumCircuit.qft() method or QFT() factory instead)
fn _qft(mut qc: QuantumCircuit, targets: List[Int], swap: Bool = True):
    """Internal helper: Apply Quantum Fourier Transform to the specified target qubits.
    """
    if swap:
        if len(targets) == qc.num_qubits:
            qc.bit_reverse()
        else:
            # Partial swap
            var n = len(targets)
            for i in range(n // 2):
                qc.swap(targets[i], targets[n - 1 - i])

    for j in range(len(targets)):
        for k in range(j):
            # cp signature: (control, target, theta)
            qc.cp(targets[j], targets[k], pi / (2 ** (j - k)))
        qc.h(targets[j])


fn _iqft(mut qc: QuantumCircuit, targets: List[Int], swap: Bool = True):
    """Internal helper: Apply Inverse Quantum Fourier Transform to the specified target qubits.
    """
    for j in reversed(range(len(targets))):
        qc.h(targets[j])
        for k in reversed(range(j)):
            # cp signature: (control, target, theta)
            qc.cp(targets[j], targets[k], -pi / (2 ** (j - k)))

    if swap:
        if len(targets) == qc.num_qubits:
            qc.bit_reverse()
        else:
            # Partial swap
            var n = len(targets)
            for i in range(n // 2):
                qc.swap(targets[i], targets[n - 1 - i])


fn QFT(m: Int, reversed: Bool = False, swap: Bool = True) -> QuantumCircuit:
    """Create a new QuantumCircuit with a QFT acting on m qubits."""
    var qc = QuantumCircuit(m)
    var reg = qc.add_register("q", m)
    qc.qft(reg, reversed, swap)
    return qc^


fn IQFT(m: Int, reversed: Bool = False, swap: Bool = True) -> QuantumCircuit:
    """Create a new QuantumCircuit with an IQFT acting on m qubits."""
    var qc = QuantumCircuit(m)
    var reg = qc.add_register("q", m)
    qc.iqft(reg, reversed, swap)
    return qc^


# =============================================================================
# Standalone Executor Functions
# =============================================================================


fn execute(mut state: QuantumState, circuit: QuantumCircuit):
    """Execute a circuit on a quantum state using generic transforms.

    Args:
        state: The quantum state to transform (modified in place).
        circuit: The circuit containing transformations to apply.
    """
    # Validate that circuit qubits match state size
    var expected_size = 1 << circuit.num_qubits
    if state.size() != expected_size:
        print(
            "Error: Circuit has",
            circuit.num_qubits,
            "qubits (expects state size",
            expected_size,
            ") but state has size",
            state.size(),
        )
        return

    # Use the circuit's execute method for now (will be refactored later)
    from butterfly.core.state import (
        transform,
        c_transform,
        mc_transform_interval,
        bit_reverse_state,
    )
    from butterfly.core.circuit import (
        Transformation,
        GateTransformation,
        SingleControlGateTransformation,
        MultiControlGateTransformation,
        BitReversalTransformation,
        DiagonalTransformation,
    )

    for i in range(len(circuit.transformations)):
        var t = circuit.transformations[i]

        if t.isa[GateTransformation]():
            var g = t[GateTransformation].copy()
            # print("Applying gate:", g.name, "to qubit", g.target, g.arg)
            transform(state, g.target, g.gate)
        elif t.isa[SingleControlGateTransformation]():
            # print("Applying single control gate")
            var g = t[SingleControlGateTransformation].copy()
            c_transform(state, g.control, g.target, g.gate)
        elif t.isa[MultiControlGateTransformation]():
            var g = t[MultiControlGateTransformation].copy()
            mc_transform_interval(state, g.controls, g.target, g.gate)
        elif t.isa[BitReversalTransformation]():
            bit_reverse_state(state)
        elif t.isa[DiagonalTransformation]():
            var g = t[DiagonalTransformation].copy()
            var n_shortcut = g.size if g.size > 0 else circuit.num_qubits
            var mask = (1 << n_shortcut) - 1
            for k in range(len(state)):
                var val = (k >> g.offset) & mask
                # Check if val is in items
                for p_idx in range(len(g.items)):
                    if val == g.items[p_idx]:
                        state[k] = Amplitude(-state[k].re, -state[k].im)
                        break


fn execute_simd[
    num_qubits: Int
](mut state: QuantumState, circuit: QuantumCircuit):
    """Execute a circuit on a quantum state using SIMD-optimized transforms.

    Args:
        state: The quantum state to transform (modified in place).
        circuit: The circuit containing transformations to apply.

    Parameters:
        num_qubits: Number of qubits (compile-time constant).

    Note:
        Requires compile-time known size. For runtime sizes, use execute() instead.
    """
    alias N = 1 << num_qubits

    # Validate that circuit qubits match parameter
    if circuit.num_qubits != num_qubits:
        print(
            "Error: Circuit has",
            circuit.num_qubits,
            "qubits but execute_simd[num_qubits] called with",
            num_qubits,
        )
        return

    # Use SIMD v2 executor
    execute_transformations_simd_v2[N](state, circuit.transformations)


fn run_circuit[n: Int](circuit: QuantumCircuit) -> QuantumState:
    """Convenience function: create state and execute with SIMD.

    Args:
        circuit: The circuit to execute.

    Parameters:
        n: Number of qubits (compile-time constant).

    Returns:
        The resulting quantum state after execution.

    Note:
        Requires compile-time known size. For runtime sizes, use run_circuit_generic() instead.
    """
    # Validate circuit size matches parameter
    if circuit.num_qubits != n:
        print(
            "Error: run_circuit[n] expects circuit with",
            n,
            "qubits, but got",
            circuit.num_qubits,
        )
        return QuantumState(n)  # Return empty state on error

    var state = QuantumState(n)
    execute_simd[1 << n](state, circuit)
    return state^


fn run_circuit_fused[n: Int](circuit: QuantumCircuit) -> QuantumState:
    """Convenience function: create state and execute with fusion.

    Args:
        circuit: The circuit to execute.

    Parameters:
        n: Number of qubits (compile-time constant).

    Returns:
        The resulting quantum state after execution.

    Note:
        Requires compile-time known size. For runtime sizes, use run_circuit_generic() instead.
    """
    # Validate circuit size matches parameter
    if circuit.num_qubits != n:
        print(
            "Error: run_circuit_fused[n] expects circuit with",
            n,
            "qubits, but got",
            circuit.num_qubits,
        )
        return QuantumState(n)  # Return empty state on error

    var state = QuantumState(n)
    from butterfly.core.execute_fused_v3 import execute_fused_v3

    execute_fused_v3[1 << n](state, circuit)
    return state^


fn run_circuit_generic(circuit: QuantumCircuit) -> QuantumState:
    """Convenience function: create state and execute with generic (non-SIMD) executor.

    Args:
        circuit: The circuit to execute.

    Returns:
        The resulting quantum state after execution.

    Note:
        Works with runtime-determined circuit sizes (no compile-time parameter needed).
        For better performance with compile-time known sizes, use run_circuit[n]() instead.
    """
    var state = QuantumState(circuit.num_qubits)
    execute(state, circuit)
    return state^


fn execute_fused[N: Int](mut state: QuantumState, circuit: QuantumCircuit):
    """Execute circuit with basic fusion (Radix-2).

    Args:
        state: The quantum state to transform (modified in place).
        circuit: The circuit containing transformations to apply.

    Parameters:
        N: Compile-time state size (must equal 1 << circuit.num_qubits).

    Note:
        For now, delegates to execute_fused_v3. Can be optimized separately later.
    """
    # Validate that circuit qubits match state size
    var expected_size = 1 << circuit.num_qubits
    if state.size() != expected_size:
        print(
            "Error: Circuit has",
            circuit.num_qubits,
            "qubits (expects state size",
            expected_size,
            ") but state has size",
            state.size(),
        )
        return

    # Delegate to fused_v3 for now
    from butterfly.core.execute_fused_v3 import execute_fused_v3

    execute_fused_v3[N](state, circuit)


fn execute_simd_v2[
    num_qubits: Int
](mut state: QuantumState, circuit: QuantumCircuit):
    """Execute circuit with SIMD v2 optimizations (optimized indexing + chunked kernels).

    Args:
        state: The quantum state to transform (modified in place).
        circuit: The circuit containing transformations to apply.

    Parameters:
        num_qubits: Number of qubits (compile-time constant).

    Note:
        Requires compile-time known size. For runtime sizes, use execute() instead.
    """
    alias N = 1 << num_qubits

    # Validate that circuit qubits match parameter
    if circuit.num_qubits != num_qubits:
        print(
            "Error: Circuit has",
            circuit.num_qubits,
            "qubits but execute_simd_v2[num_qubits] called with",
            num_qubits,
        )
        return

    # Use SIMD v2 executor (unfused)
    execute_transformations_simd_v2[N](state, circuit.transformations)


fn execute_simd_unfused[
    N: Int
](mut state: QuantumState, circuit: QuantumCircuit):
    """Execute circuit with SIMD but no fusion.

    Args:
        state: The quantum state to transform (modified in place).
        circuit: The circuit containing transformations to apply.

    Parameters:
        N: Compile-time state size (must equal 1 << circuit.num_qubits).

    Note:
        Same as execute_simd - unfused SIMD execution.
    """
    # Validate that circuit qubits match state size
    var expected_size = 1 << circuit.num_qubits
    if state.size() != expected_size:
        print(
            "Error: Circuit has",
            circuit.num_qubits,
            "qubits (expects state size",
            expected_size,
            ") but state has size",
            state.size(),
        )
        return

    # Use SIMD v2 executor (unfused)
    execute_transformations_simd_v2[N](state, circuit.transformations)


fn execute_fused_v3[
    num_qubits: Int
](mut state: QuantumState, circuit: QuantumCircuit):
    """Execute circuit with fused_v3 optimization (fusion + matrix multiplication).

    Args:
        state: The quantum state to transform (modified in place).
        circuit: The circuit containing transformations to apply.

    Parameters:
        num_qubits: Number of qubits (compile-time constant).

    Note:
        Delegates to execute_fused_v3 implementation for best performance.
    """
    alias N = 1 << num_qubits

    # Validate that circuit qubits match parameter
    if circuit.num_qubits != num_qubits:
        print(
            "Error: Circuit has",
            circuit.num_qubits,
            "qubits but execute_fused_v3[num_qubits] called with",
            num_qubits,
        )
        return

    # Delegate to fused_v3 for now (best available optimizer)
    from butterfly.core.execute_fused_v3 import execute_fused_v3

    execute_fused_v3[N](state, circuit)


fn run_circuit_optimized[n: Int](circuit: QuantumCircuit) -> QuantumState:
    """Convenience function: create state and execute with optimization.

    Args:
        circuit: The circuit to execute.

    Parameters:
        n: Number of qubits (compile-time constant).

    Returns:
        The resulting quantum state after execution.
    """
    # Validate circuit size matches parameter
    if circuit.num_qubits != n:
        print(
            "Error: run_circuit_optimized[n] expects circuit with",
            n,
            "qubits, but got",
            circuit.num_qubits,
        )
        return QuantumState(n)

    var state = QuantumState(n)
    execute_fused_v3[1 << n](state, circuit)
    return state^


fn execute_with_strategy[
    n: Int, strategy: ExecutionStrategy
](mut state: QuantumState, circuit: QuantumCircuit):
    """Execute circuit with specified strategy (fully compile-time specialized).

    Args:
        state: The quantum state to transform (modified in place).
        circuit: The circuit containing transformations to apply.

    Parameters:
        n: Number of qubits (compile-time constant).
        strategy: Execution strategy to use (compile-time constant).

    Note:
        This is the fully compile-time specialized version with zero runtime overhead.
        Both n and strategy are known at compile time, allowing maximum optimization.
        For runtime strategy selection, use the overload with strategy as an argument.
    """
    # Validate circuit size matches parameter
    if circuit.num_qubits != n:
        print(
            "Error: execute_with_strategy[n, strategy] expects circuit with",
            n,
            "qubits, but got",
            circuit.num_qubits,
        )
        return

    # Compile-time dispatch - no runtime branching!
    @parameter
    if strategy.isa[Generic]():
        execute_simd[n](state, circuit)
    elif strategy.isa[SIMD]():
        execute_simd[n](state, circuit)
    elif strategy.isa[SIMDv2]():
        execute_simd_v2[n](state, circuit)
    elif strategy.isa[FusedV3]():
        execute_fused_v3[n](state, circuit)
    else:
        execute_simd[n](state, circuit)


fn execute_with_strategy[
    n: Int
](
    mut state: QuantumState,
    circuit: QuantumCircuit,
    strategy: ExecutionStrategy,
):
    """Execute circuit with specified strategy (compile-time specialized).

    Args:
        state: The quantum state to transform (modified in place).
        circuit: The circuit containing transformations to apply.
        strategy: Execution strategy to use.

    Parameters:
        n: Number of qubits (compile-time constant).

    Note:
        This is the compile-time specialized version that dispatches to the
        appropriate executor based on the strategy. For dynamic dispatch, use
        execute_with_strategy_dynamic().
    """
    alias N = 1 << n

    # Validate circuit size matches parameter
    if circuit.num_qubits != n:
        print(
            "Error: execute_with_strategy[n] expects circuit with",
            n,
            "qubits, but got",
            circuit.num_qubits,
        )
        return

    # Dispatch to appropriate executor based on strategy
    if strategy.isa[Generic]():
        execute_simd[n](state, circuit)  # Generic uses SIMD implementation
    elif strategy.isa[SIMD]():
        execute_simd[n](state, circuit)
    elif strategy.isa[SIMDv2]():
        execute_simd_v2[n](state, circuit)
    elif strategy.isa[FusedV3]():
        execute_fused_v3[n](state, circuit)
    else:
        execute_simd[n](state, circuit)  # fallback to SIMD
