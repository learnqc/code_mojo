from butterfly.core.state import (
    QuantumState,
    transform,
    c_transform,
    mc_transform_interval,
    bit_reverse_state,
)
from butterfly.core.types import *
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


struct FusedTransformation(Copyable, Movable):
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
        var r4 = InlineArray[Amplitude, 4](Amplitude(0, 0))
        self.m4 = Matrix4x4(r4, r4, r4, r4)
        self.m8 = mat
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

    fn __copyinit__(out self, existing: Self):
        self.type = existing.type
        self.q0 = existing.q0
        self.q1 = existing.q1
        self.q2 = existing.q2
        self.q3 = existing.q3
        self.m4 = existing.m4
        self.m8 = existing.m8
        self.m16 = existing.m16

    fn __moveinit__(out self, deinit existing: Self):
        self.type = existing.type
        self.q0 = existing.q0
        self.q1 = existing.q1
        self.q2 = existing.q2
        self.q3 = existing.q3
        self.m4 = existing.m4
        self.m8 = existing.m8
        self.m16 = existing.m16


struct QuantumTransformation(Copyable, Movable):
    """A quantum transformation representing a gate applied to target qubit(s) with optional controls.
    """

    var gate: Gate
    var target: Int
    var controls: List[Int]
    var is_permutation: Bool

    fn __init__(out self, gate: Gate, target: Int):
        self.gate = gate
        self.target = target
        self.controls = List[Int]()
        self.is_permutation = False

    fn __init__(out self, gate: Gate, target: Int, var controls: List[Int]):
        self.gate = gate
        self.target = target
        self.controls = controls^
        self.is_permutation = False

    fn __init__(out self, is_permutation: Bool):
        self.gate = X  # Dummy
        self.target = -1  # Dummy
        self.controls = List[Int]()
        self.is_permutation = is_permutation

    fn __copyinit__(out self, existing: Self):
        self.gate = existing.gate
        self.target = existing.target
        self.controls = List[Int](capacity=len(existing.controls))
        for q in existing.controls:
            self.controls.append(q)
        self.is_permutation = existing.is_permutation

    fn __moveinit__(out self, deinit existing: Self):
        self.gate = existing.gate
        self.target = existing.target
        self.controls = existing.controls^
        self.is_permutation = existing.is_permutation

    fn is_controlled(self) -> Bool:
        return len(self.controls) > 0

    fn num_controls(self) -> Int:
        return len(self.controls)

    fn add_control(mut self, q: Int):
        self.controls.append(q)

    fn get_involved_qubits(self) -> List[Int]:
        """Return a list of all qubits involved in this transformation."""
        var res = List[Int]()
        if self.is_permutation:
            return res^
        res.append(self.target)
        for q in self.controls:
            res.append(q)
        return res^

    fn get_as_matrix4x4(self, q_high: Int, q_low: Int) -> Matrix4x4:
        """Get the 4x4 matrix representation of this transformation with respect to two qubits.
        """
        var identity2 = Gate(
            InlineArray[Amplitude, 2](Amplitude(1, 0), Amplitude(0, 0)),
            InlineArray[Amplitude, 2](Amplitude(0, 0), Amplitude(1, 0)),
        )

        if self.is_permutation:
            return compute_kron_product(identity2, identity2)

        if not self.is_controlled():
            if self.target == q_high:
                return compute_kron_product(self.gate, identity2)
            elif self.target == q_low:
                return compute_kron_product(identity2, self.gate)
            else:
                return compute_kron_product(identity2, identity2)

        var t = self.target
        var row = InlineArray[Amplitude, 4](
            Amplitude(0, 0), Amplitude(0, 0), Amplitude(0, 0), Amplitude(0, 0)
        )
        var m = Matrix4x4(row, row, row, row)

        for i in range(4):
            var b_high = (i >> 1) & 1
            var b_low = i & 1

            var controls_satisfied = True
            for q_idx in range(self.num_controls()):
                var ctrl = self.controls[q_idx]
                var val: Int
                if ctrl == q_high:
                    val = b_high
                elif ctrl == q_low:
                    val = b_low
                else:
                    # Control is outside this 2-qubit subspace, which should not happen
                    # if the caller ensured all qubits fit.
                    controls_satisfied = False
                    break
                if val == 0:
                    controls_satisfied = False
                    break

            if not controls_satisfied:
                m[i][i] = Amplitude(1, 0)
            else:
                var b_t: Int
                if t == q_high:
                    b_t = b_high
                else:
                    b_t = b_low

                m[i][i] = self.gate[b_t][b_t]
                var flipped_j = i ^ (2 if t == q_high else 1)
                m[flipped_j][i] = self.gate[1 - b_t][b_t]

        return m

    fn get_as_matrix8x8(self, q_high: Int, q_mid: Int, q_low: Int) -> Matrix8x8:
        """Get the 8x8 matrix representation of this transformation with respect to three qubits.
        """
        var identity2 = Gate(
            InlineArray[Amplitude, 2](Amplitude(1, 0), Amplitude(0, 0)),
            InlineArray[Amplitude, 2](Amplitude(0, 0), Amplitude(1, 0)),
        )

        # Helper to create an 8x8 identity
        var row0 = InlineArray[Amplitude, 8](
            Amplitude(1, 0),
            Amplitude(0, 0),
            Amplitude(0, 0),
            Amplitude(0, 0),
            Amplitude(0, 0),
            Amplitude(0, 0),
            Amplitude(0, 0),
            Amplitude(0, 0),
        )
        var row1 = InlineArray[Amplitude, 8](
            Amplitude(0, 0),
            Amplitude(1, 0),
            Amplitude(0, 0),
            Amplitude(0, 0),
            Amplitude(0, 0),
            Amplitude(0, 0),
            Amplitude(0, 0),
            Amplitude(0, 0),
        )
        var row2 = InlineArray[Amplitude, 8](
            Amplitude(0, 0),
            Amplitude(0, 0),
            Amplitude(1, 0),
            Amplitude(0, 0),
            Amplitude(0, 0),
            Amplitude(0, 0),
            Amplitude(0, 0),
            Amplitude(0, 0),
        )
        var row3 = InlineArray[Amplitude, 8](
            Amplitude(0, 0),
            Amplitude(0, 0),
            Amplitude(0, 0),
            Amplitude(1, 0),
            Amplitude(0, 0),
            Amplitude(0, 0),
            Amplitude(0, 0),
            Amplitude(0, 0),
        )
        var row4 = InlineArray[Amplitude, 8](
            Amplitude(0, 0),
            Amplitude(0, 0),
            Amplitude(0, 0),
            Amplitude(0, 0),
            Amplitude(1, 0),
            Amplitude(0, 0),
            Amplitude(0, 0),
            Amplitude(0, 0),
        )
        var row5 = InlineArray[Amplitude, 8](
            Amplitude(0, 0),
            Amplitude(0, 0),
            Amplitude(0, 0),
            Amplitude(0, 0),
            Amplitude(0, 0),
            Amplitude(1, 0),
            Amplitude(0, 0),
            Amplitude(0, 0),
        )
        var row6 = InlineArray[Amplitude, 8](
            Amplitude(0, 0),
            Amplitude(0, 0),
            Amplitude(0, 0),
            Amplitude(0, 0),
            Amplitude(0, 0),
            Amplitude(0, 0),
            Amplitude(1, 0),
            Amplitude(0, 0),
        )
        var row7 = InlineArray[Amplitude, 8](
            Amplitude(0, 0),
            Amplitude(0, 0),
            Amplitude(0, 0),
            Amplitude(0, 0),
            Amplitude(0, 0),
            Amplitude(0, 0),
            Amplitude(0, 0),
            Amplitude(1, 0),
        )
        var id8 = Matrix8x8(row0, row1, row2, row3, row4, row5, row6, row7)

        if self.is_permutation:
            return id8

        if not self.is_controlled():
            if self.target == q_high:
                # G x I2 x I2
                var m4 = compute_kron_product(self.gate, identity2)
                return self._kron4_2(m4, identity2)
            elif self.target == q_mid:
                # I2 x G x I2
                var m4 = compute_kron_product(identity2, self.gate)
                return self._kron4_2(m4, identity2)
            elif self.target == q_low:
                # I2 x I2 x G
                var m4 = compute_kron_product(identity2, identity2)
                return self._kron4_2(m4, self.gate)
            else:
                return id8

        var t = self.target
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
        var m = Matrix8x8(row, row, row, row, row, row, row, row)

        for i in range(8):
            var b_high = (i >> 2) & 1
            var b_mid = (i >> 1) & 1
            var b_low = i & 1

            var controls_satisfied = True
            for q_idx in range(self.num_controls()):
                var ctrl = self.controls[q_idx]
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
                if t == q_high:
                    b_t = b_high
                elif t == q_mid:
                    b_t = b_mid
                else:
                    b_t = b_low

                m[i][i] = self.gate[b_t][b_t]
                var flipped_j: Int
                if t == q_high:
                    flipped_j = i ^ 4
                elif t == q_mid:
                    flipped_j = i ^ 2
                else:
                    flipped_j = i ^ 1
                m[flipped_j][i] = self.gate[1 - b_t][b_t]
        return m

    fn get_as_matrix16x16(
        self, q3: Int, q2: Int, q1: Int, q0: Int
    ) -> Matrix16x16:
        """Get the 16x16 matrix representation of this transformation with respect to four qubits.
        q3 > q2 > q1 > q0.
        """
        var row = InlineArray[Amplitude, 16](
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
            row,
            row,
            row,
            row,
            row,
            row,
            row,
            row,
            row,
            row,
            row,
            row,
            row,
            row,
            row,
            row,
        )
        for i in range(16):
            id16[i][i] = Amplitude(1, 0)

        if self.is_permutation:
            return id16

        if not self.is_controlled():
            var t = self.target
            var m = Matrix16x16(
                row,
                row,
                row,
                row,
                row,
                row,
                row,
                row,
                row,
                row,
                row,
                row,
                row,
                row,
                row,
                row,
            )
            for i in range(16):
                var b3 = (i >> 3) & 1
                var b2 = (i >> 2) & 1
                var b1 = (i >> 1) & 1
                var b0 = i & 1
                var b_t: Int
                if t == q3:
                    b_t = b3
                elif t == q2:
                    b_t = b2
                elif t == q1:
                    b_t = b1
                else:
                    b_t = b0
                m[i][i] = self.gate[b_t][b_t]
                var flipped_j: Int
                if t == q3:
                    flipped_j = i ^ 8
                elif t == q2:
                    flipped_j = i ^ 4
                elif t == q1:
                    flipped_j = i ^ 2
                else:
                    flipped_j = i ^ 1
                m[flipped_j][i] = self.gate[1 - b_t][b_t]
            return m

        var t = self.target
        var m = Matrix16x16(
            row,
            row,
            row,
            row,
            row,
            row,
            row,
            row,
            row,
            row,
            row,
            row,
            row,
            row,
            row,
            row,
        )
        for i in range(16):
            var b3 = (i >> 3) & 1
            var b2 = (i >> 2) & 1
            var b1 = (i >> 1) & 1
            var b0 = i & 1

            var controls_satisfied = True
            for q_idx in range(self.num_controls()):
                var ctrl = self.controls[q_idx]
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
                if t == q3:
                    b_t = b3
                elif t == q2:
                    b_t = b2
                elif t == q1:
                    b_t = b1
                else:
                    b_t = b0

                m[i][i] = self.gate[b_t][b_t]
                var flipped_j: Int
                if t == q3:
                    flipped_j = i ^ 8
                elif t == q2:
                    flipped_j = i ^ 4
                elif t == q1:
                    flipped_j = i ^ 2
                else:
                    flipped_j = i ^ 1
                m[flipped_j][i] = self.gate[1 - b_t][b_t]
        return m

    fn _kron4_2(self, m4: Matrix4x4, m2: Gate) -> Matrix8x8:
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


struct QuantumCircuit(Copyable):
    """A quantum circuit that manages quantum state and transformation operations.

    The circuit maintains a quantum state and a sequence of transformations (gates)
    that can be applied to the state. Supports quantum registers for organizing qubits.
    """

    var state: QuantumState
    var transformations: List[QuantumTransformation]
    var num_qubits: Int
    var registers: List[QuantumRegister]
    var fused_transformations: List[FusedTransformation]
    var is_fused: Bool

    fn __init__(out self, num_qubits: Int):
        """Initialize a quantum circuit with n qubits in the |0⟩ state."""
        self.num_qubits = num_qubits
        self.state = QuantumState(num_qubits)
        self.transformations = List[QuantumTransformation]()
        self.registers = List[QuantumRegister]()
        self.fused_transformations = List[FusedTransformation]()
        self.is_fused = False

    fn __init__(out self, var state: QuantumState, num_qubits: Int):
        """Initialize a quantum circuit with a given state."""
        self.num_qubits = num_qubits
        self.state = state^
        self.transformations = List[QuantumTransformation]()
        self.registers = List[QuantumRegister]()
        self.fused_transformations = List[FusedTransformation]()
        self.is_fused = False

    fn __copyinit__(out self, existing: Self):
        self.num_qubits = existing.num_qubits
        self.state = existing.state.copy()
        self.transformations = List[QuantumTransformation](
            capacity=len(existing.transformations)
        )
        for i in range(len(existing.transformations)):
            self.transformations.append(existing.transformations[i].copy())
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
        self.is_fused = existing.is_fused

    fn __moveinit__(out self, deinit existing: Self):
        self.num_qubits = existing.num_qubits
        self.state = existing.state^
        self.transformations = existing.transformations^
        self.registers = existing.registers^
        self.fused_transformations = existing.fused_transformations^
        self.is_fused = existing.is_fused

    fn add(mut self, gate: Gate, target: Int):
        """Add a gate to the circuit on the specified target qubit."""
        self.is_fused = False
        var transformation = QuantumTransformation(gate, target)
        self.transformations.append(transformation^)

    fn add_controlled(mut self, gate: Gate, target: Int, control: Int):
        """Add a controlled gate to the circuit."""
        var controls = List[Int]()
        controls.append(control)
        var transformation = QuantumTransformation(gate, target, controls^)
        self.transformations.append(transformation^)

    fn add_multi_controlled(
        mut self, gate: Gate, target: Int, var controls: List[Int]
    ):
        """Add a multi-controlled gate to the circuit."""
        var transformation = QuantumTransformation(gate, target, controls^)
        self.transformations.append(transformation^)

    fn apply_transformation(mut self, t: QuantumTransformation):
        """Apply a single transformation to the state."""
        if t.is_permutation:
            bit_reverse_state(self.state)
            return

        if t.is_controlled():
            if t.num_controls() == 1:
                c_transform(self.state, t.controls[0], t.target, t.gate)
            else:
                mc_transform_interval(self.state, t.controls, t.target, t.gate)
        else:
            transform(self.state, t.target, t.gate)

    fn execute(mut self):
        """Execute all transformations in the circuit on the quantum state."""
        for i in range(len(self.transformations)):
            var t = self.transformations[i].copy()
            self.apply_transformation(t)

    fn execute_fused(mut self):
        """Execute pre-computed fused transformations."""
        if not self.is_fused:
            self.fuse()

        for i in range(len(self.fused_transformations)):
            var ft = self.fused_transformations[i].copy()
            if ft.type == 0:
                bit_reverse_state(self.state)
            elif ft.type == 1:
                transform_matrix4(self.state, ft.q1, ft.q0, ft.m4)
            elif ft.type == 2:
                transform_matrix8(self.state, ft.q2, ft.q1, ft.q0, ft.m8)
            elif ft.type == 3:
                transform_matrix16(
                    self.state, ft.q3, ft.q2, ft.q1, ft.q0, ft.m16
                )

    fn execute_simd(mut self):
        """
        Execute circuit with SIMD optimizations for common sizes.
        Dispatches to transform_simd[N] for N=20-30, beating Qiskit.
        """
        var n = self.state.size()
        var num_qubits = 0
        var temp = n
        while temp > 1:
            temp >>= 1
            num_qubits += 1

        # Dispatch to SIMD for common qubit counts
        from butterfly.core.execute_simd_dispatch import (
            execute_transformations_simd,
        )

        if num_qubits == 25:
            execute_transformations_simd[1 << 25](
                self.state, self.transformations
            )
        elif num_qubits == 24:
            execute_transformations_simd[1 << 24](
                self.state, self.transformations
            )
        elif num_qubits == 26:
            execute_transformations_simd[1 << 26](
                self.state, self.transformations
            )
        elif num_qubits == 23:
            execute_transformations_simd[1 << 23](
                self.state, self.transformations
            )
        elif num_qubits == 27:
            execute_transformations_simd[1 << 27](
                self.state, self.transformations
            )
        elif num_qubits == 28:
            execute_transformations_simd[1 << 28](
                self.state, self.transformations
            )
        elif num_qubits == 29:
            execute_transformations_simd[1 << 29](
                self.state, self.transformations
            )
        elif num_qubits == 30:
            execute_transformations_simd[1 << 30](
                self.state, self.transformations
            )
        elif num_qubits == 22:
            execute_transformations_simd[1 << 22](
                self.state, self.transformations
            )
        elif num_qubits == 21:
            execute_transformations_simd[1 << 21](
                self.state, self.transformations
            )
        elif num_qubits == 20:
            execute_transformations_simd[1 << 20](
                self.state, self.transformations
            )
        elif num_qubits == 19:
            execute_transformations_simd[1 << 19](
                self.state, self.transformations
            )
        elif num_qubits == 18:
            execute_transformations_simd[1 << 18](
                self.state, self.transformations
            )
        elif num_qubits == 17:
            execute_transformations_simd[1 << 17](
                self.state, self.transformations
            )
        elif num_qubits == 16:
            execute_transformations_simd[1 << 16](
                self.state, self.transformations
            )
        elif num_qubits == 15:
            execute_transformations_simd[1 << 15](
                self.state, self.transformations
            )
        elif num_qubits == 14:
            execute_transformations_simd[1 << 14](
                self.state, self.transformations
            )
        elif num_qubits == 13:
            execute_transformations_simd[1 << 13](
                self.state, self.transformations
            )
        elif num_qubits == 12:
            execute_transformations_simd[1 << 12](
                self.state, self.transformations
            )
        elif num_qubits == 11:
            execute_transformations_simd[1 << 11](
                self.state, self.transformations
            )
        elif num_qubits == 10:
            execute_transformations_simd[1 << 10](
                self.state, self.transformations
            )
        else:
            # Fall back to standard execution for other sizes
            self.execute()

    fn execute_simd_v2(mut self):
        """
        Execute circuit with SIMD optimizations v2.
        Optimized indexing and chunked kernels.
        """
        var n = self.state.size()
        var num_qubits = 0
        var temp = n
        while temp > 1:
            temp >>= 1
            num_qubits += 1

        # Dispatch to SIMD v2 for common qubit counts
        from butterfly.core.execute_simd_v2_dispatch import (
            execute_transformations_simd_v2,
        )

        if num_qubits == 25:
            execute_transformations_simd_v2[1 << 25](
                self.state, self.transformations
            )
        elif num_qubits == 24:
            execute_transformations_simd_v2[1 << 24](
                self.state, self.transformations
            )
        elif num_qubits == 26:
            execute_transformations_simd_v2[1 << 26](
                self.state, self.transformations
            )
        elif num_qubits == 23:
            execute_transformations_simd_v2[1 << 23](
                self.state, self.transformations
            )
        elif num_qubits == 27:
            execute_transformations_simd_v2[1 << 27](
                self.state, self.transformations
            )
        elif num_qubits == 28:
            execute_transformations_simd_v2[1 << 28](
                self.state, self.transformations
            )
        elif num_qubits == 29:
            execute_transformations_simd_v2[1 << 29](
                self.state, self.transformations
            )
        elif num_qubits == 30:
            execute_transformations_simd_v2[1 << 30](
                self.state, self.transformations
            )
        elif num_qubits == 22:
            execute_transformations_simd_v2[1 << 22](
                self.state, self.transformations
            )
        elif num_qubits == 21:
            execute_transformations_simd_v2[1 << 21](
                self.state, self.transformations
            )
        elif num_qubits == 20:
            execute_transformations_simd_v2[1 << 20](
                self.state, self.transformations
            )
        elif num_qubits == 19:
            execute_transformations_simd_v2[1 << 19](
                self.state, self.transformations
            )
        elif num_qubits == 18:
            execute_transformations_simd_v2[1 << 18](
                self.state, self.transformations
            )
        elif num_qubits == 17:
            execute_transformations_simd_v2[1 << 17](
                self.state, self.transformations
            )
        elif num_qubits == 16:
            execute_transformations_simd_v2[1 << 16](
                self.state, self.transformations
            )
        elif num_qubits == 15:
            execute_transformations_simd_v2[1 << 15](
                self.state, self.transformations
            )
        elif num_qubits == 14:
            execute_transformations_simd_v2[1 << 14](
                self.state, self.transformations
            )
        elif num_qubits == 13:
            execute_transformations_simd_v2[1 << 13](
                self.state, self.transformations
            )
        elif num_qubits == 12:
            execute_transformations_simd_v2[1 << 12](
                self.state, self.transformations
            )
        elif num_qubits == 11:
            execute_transformations_simd_v2[1 << 11](
                self.state, self.transformations
            )
        elif num_qubits == 10:
            execute_transformations_simd_v2[1 << 10](
                self.state, self.transformations
            )
        else:
            # Fall back to execute_simd() if v2 not available or requested
            self.execute_simd()

    fn apply_transformation_super_fast(mut self, t: QuantumTransformation):
        """
        Apply transformation with all optimizations enabled.
        Uses specialized SIMD kernels for common gates.
        """
        if t.is_permutation:
            bit_reverse_state(self.state)
            return

        if t.is_controlled():
            # Use optimized controlled-H if applicable
            if t.num_controls() == 1 and is_h(t.gate):
                from butterfly.core.c_transform_fast import c_transform_h_simd

                c_transform_h_simd(self.state, t.controls[0], t.target)
            elif t.num_controls() == 1:
                c_transform(self.state, t.controls[0], t.target, t.gate)
            else:
                mc_transform_interval(self.state, t.controls, t.target, t.gate)
        else:
            # Use optimized single-qubit gates
            if is_h(t.gate):
                from butterfly.core.state import transform_h_block_style

                transform_h_block_style(self.state, t.target)
            else:
                transform(self.state, t.target, t.gate)

    fn execute_simd_unfused(mut self):
        """
        Execute with optimizations but without fusion.
        Useful for comparing optimization impact vs fusion impact.
        """
        for i in range(len(self.transformations)):
            var t = self.transformations[i].copy()
            self.apply_transformation_super_fast(t)

    fn fuse(mut self):
        """Perform greedy fusion and pre-compute transformation matrices."""
        if self.is_fused:
            return

        self.fused_transformations = List[FusedTransformation]()
        var i = 0
        var n_ops = len(self.transformations)

        while i < n_ops:
            var t = self.transformations[i].copy()

            if t.is_permutation:
                self.fused_transformations.append(
                    FusedTransformation(is_perm=True)
                )
                i += 1
                continue

            var q_set = t.get_involved_qubits()
            var j = i + 1
            while j < n_ops:
                var nt = self.transformations[j].copy()
                if nt.is_permutation:
                    break

                var nq = nt.get_involved_qubits()
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

                var mat = t.get_as_matrix4x4(q_high, q_low)
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

                var fused_mat = (
                    self.transformations[i]
                    .copy()
                    .get_as_matrix16x16(q3, q2, q1, q0)
                )
                for k in range(i + 1, j):
                    var next_mat = (
                        self.transformations[k]
                        .copy()
                        .get_as_matrix16x16(q3, q2, q1, q0)
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

                var fused_mat = (
                    self.transformations[i]
                    .copy()
                    .get_as_matrix8x8(q_high, q_mid, q_low)
                )
                for k in range(i + 1, j):
                    var next_mat = (
                        self.transformations[k]
                        .copy()
                        .get_as_matrix8x8(q_high, q_mid, q_low)
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

                var fused_mat = (
                    self.transformations[i]
                    .copy()
                    .get_as_matrix4x4(q_high, q_low)
                )
                for k in range(i + 1, j):
                    var next_mat = (
                        self.transformations[k]
                        .copy()
                        .get_as_matrix4x4(q_high, q_low)
                    )
                    fused_mat = matmul_matrix4x4(next_mat, fused_mat)

                self.fused_transformations.append(
                    FusedTransformation(q_high, q_low, fused_mat)
                )
                i = j

        self.is_fused = True

    fn execute_optimized(mut self):
        """Execute transformations with automatic fusion optimization."""
        if not self.is_fused:
            self.fuse()
        self.execute_fused()

    fn clear_transformations(mut self):
        """Clear all transformations from the circuit."""
        self.transformations = List[QuantumTransformation]()

    fn num_transformations(self) -> Int:
        """Return the number of transformations in the circuit."""
        return len(self.transformations)

    fn get_state(self) -> QuantumState:
        """Return a copy of the current quantum state."""
        return self.state.copy()

    fn get_amplitude(self, idx: Int) -> Amplitude:
        """Get the amplitude at a specific basis state index."""
        return self.state[idx]

    fn set_amplitude(mut self, idx: Int, amp: Amplitude):
        """Set the amplitude at a specific basis state index."""
        self.state[idx] = amp

    # Gate-specific methods
    fn h(mut self, target: Int):
        """Apply Hadamard gate to target qubit."""
        self.add(H, target)

    fn x(mut self, target: Int):
        """Apply Pauli-X gate to target qubit."""
        self.add(X, target)

    fn y(mut self, target: Int):
        """Apply Pauli-Y gate to target qubit."""
        self.add(Y, target)

    fn z(mut self, target: Int):
        """Apply Pauli-Z gate to target qubit."""
        self.add(Z, target)

    fn rx(mut self, target: Int, theta: FloatType):
        """Apply RX rotation gate to target qubit."""
        self.add(RX(theta), target)

    fn ry(mut self, target: Int, theta: FloatType):
        """Apply RY rotation gate to target qubit."""
        self.add(RY(theta), target)

    fn rz(mut self, target: Int, theta: FloatType):
        """Apply RZ rotation gate to target qubit."""
        self.add(RZ(theta), target)

    fn p(mut self, target: Int, theta: FloatType):
        """Apply phase gate to target qubit."""
        self.add(P(theta), target)

    # Controlled versions
    fn ch(mut self, target: Int, control: Int):
        """Apply controlled Hadamard gate."""
        self.add_controlled(H, target, control)

    fn cx(mut self, target: Int, control: Int):
        """Apply controlled X (CNOT) gate."""
        self.add_controlled(X, target, control)

    fn cy(mut self, target: Int, control: Int):
        """Apply controlled Y gate."""
        self.add_controlled(Y, target, control)

    fn cz(mut self, target: Int, control: Int):
        """Apply controlled Z gate."""
        self.add_controlled(Z, target, control)

    fn crx(mut self, target: Int, control: Int, theta: FloatType):
        """Apply controlled RX rotation gate."""
        self.add_controlled(RX(theta), target, control)

    fn cry(mut self, target: Int, control: Int, theta: FloatType):
        """Apply controlled RY rotation gate."""
        self.add_controlled(RY(theta), target, control)

    fn crz(mut self, target: Int, control: Int, theta: FloatType):
        """Apply controlled RZ rotation gate."""
        self.add_controlled(RZ(theta), target, control)

    fn cp(mut self, target: Int, control: Int, theta: FloatType):
        """Apply controlled phase gate."""
        self.add_controlled(P(theta), target, control)

    fn mcx(mut self, var controls: List[Int], target: Int):
        """Apply multi-controlled X gate."""
        self.add_multi_controlled(X, target, controls^)

    fn swap(mut self, q1: Int, q2: Int):
        """Apply SWAP gate between two qubits using 3 CNOTs."""
        self.cx(q2, q1)
        self.cx(q1, q2)
        self.cx(q2, q1)

    fn bit_reverse(mut self):
        """Add an efficient bit-reversal operation to the circuit."""
        self.is_fused = False
        var t = QuantumTransformation(is_permutation=True)
        self.transformations.append(t^)

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


alias Circuit = QuantumCircuit
alias Register = QuantumRegister
alias Transformation = QuantumTransformation
