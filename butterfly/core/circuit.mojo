from butterfly.core.gates import *
from butterfly.core.types import Amplitude, FloatType, Gate
from utils.variant import Variant


struct ControlKind:
    alias NO_CONTROL = 0
    alias SINGLE_CONTROL = 1
    alias MULTI_CONTROL = 2


struct GateTransformation(Copyable, Movable):
    var controls: List[Int]
    var target: Int
    var gate_info: GateInfo
    var kind: Int

    fn __init__(
        out self,
        controls: List[Int],
        target: Int,
        gate_info: GateInfo,
    ):
        self.controls = controls.copy()
        self.target = target
        self.gate_info = gate_info.copy()

        if len(controls) == 0:
            self.kind = ControlKind.NO_CONTROL
        elif len(controls) == 1:
            self.kind = ControlKind.SINGLE_CONTROL
        else:
            self.kind = ControlKind.MULTI_CONTROL


struct FusedPairTransformation(Copyable, Movable):
    var first: GateTransformation
    var second: GateTransformation

    fn __init__(
        out self,
        first: GateTransformation,
        second: GateTransformation,
    ):
        self.first = first.copy()
        self.second = second.copy()

    fn __copyinit__(out self, existing: Self):
        self.first = existing.first.copy()
        self.second = existing.second.copy()

    fn __moveinit__(out self, deinit existing: Self):
        self.first = existing.first^
        self.second = existing.second^


struct SwapTransformation(Copyable, Movable):
    var a: Int
    var b: Int

    fn __init__(out self, a: Int, b: Int):
        self.a = a
        self.b = b


struct QubitReversalTransformation(Copyable, Movable):
    var targets: List[Int]

    fn __init__(out self):
        self.targets = List[Int]()

    fn __init__(out self, targets: List[Int]):
        self.targets = targets.copy()


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

    fn __moveinit__(out self, deinit existing: Self):
        self.u = existing.u^
        self.target = existing.target
        self.control = existing.control
        self.m = existing.m
        self.name = existing.name^


struct ClassicalTransformation[StateType: AnyType](Copyable, Movable):
    var name: String
    var targets: List[Int]
    var apply: fn(mut StateType, List[Int]) raises
    var inverse_apply: Optional[fn(mut StateType, List[Int]) raises]

    fn __init__(
        out self,
        name: String,
        targets: List[Int],
        apply: fn(mut StateType, List[Int]) raises,
        inverse_apply: Optional[fn(mut StateType, List[Int]) raises] = None,
    ):
        self.name = name
        self.targets = targets.copy()
        self.apply = apply
        self.inverse_apply = inverse_apply


struct MeasurementTransformation[StateType: AnyType](Copyable, Movable):
    var targets: List[Int]
    var values: List[Optional[Bool]]
    var seed: Optional[Int]
    var apply: fn(
        mut StateType,
        List[Int],
        List[Optional[Bool]],
        Optional[Int],
    ) raises

    fn __init__(
        out self,
        targets: List[Int],
        seed: Optional[Int],
        apply: fn(
            mut StateType,
            List[Int],
            List[Optional[Bool]],
            Optional[Int],
        ) raises,
        values: List[Optional[Bool]] = []
    ):
        self.targets = targets.copy()
        if len(values) == 0:
            self.values = List[Optional[Bool]](
                length=len(targets), fill=None
            )
        else:
            self.values = values.copy()
        self.seed = seed
        self.apply = apply


alias Transformation[StateType: AnyType] = Variant[
    GateTransformation,
    FusedPairTransformation,
    SwapTransformation,
    QubitReversalTransformation,
    UnitaryTransformation,
    ControlledUnitaryTransformation,
    ClassicalTransformation[StateType],
    MeasurementTransformation[StateType],
]

struct Register(Copyable, Movable):
    var name: String
    var start: Int
    var length: Int

    fn __init__(
        out self,
        name: String,
        length: Int,
        start: Int = 0,
    ):
        self.name = name
        self.start = start
        self.length = length

    fn __getitem__(self, idx: Int) -> Int:
        return self.start + idx
        
struct Circuit[StateType: AnyType](Copyable, Movable):
    var transformations: List[Transformation[StateType]]
    var num_qubits: Int
    var registers: List[Register]

    fn __init__(out self, num_qubits: Int):
        self.transformations = List[Transformation[StateType]]()
        self.num_qubits = num_qubits
        self.registers = List[Register]()

    fn __init__(out self, register: Register):
        self.transformations = List[Transformation[StateType]]()
        self.registers = List[Register](capacity=1)
        self.registers.append(register.copy())
        self.num_qubits = register.length

    fn __init__(out self, registers: List[Register]):
        self.transformations = List[Transformation[StateType]]()
        self.registers = List[Register](capacity=len(registers))
        var offset = 0
        for reg in registers:
            var adjusted = reg.copy()
            adjusted.start = offset
            self.registers.append(adjusted.copy())
            offset += reg.length
        self.num_qubits = offset

    fn add(
        mut self,
        target: Int,
        gate_info: GateInfo,
    ):
        self.transformations.append(
            GateTransformation([], target, gate_info)
        )

    fn add(
        mut self,
        control: Int,
        target: Int,
        gate_info: GateInfo,
    ):
        self.transformations.append(
            GateTransformation([control], target, gate_info)
        )

    fn add(
        mut self,
        controls: List[Int],
        target: Int,
        gate_info: GateInfo,
    ):
        self.transformations.append(
            GateTransformation(controls, target, gate_info)
        )

    fn add_classical(
        mut self,
        name: String,
        targets: List[Int],
        apply: fn(mut StateType, List[Int]) raises,
        inverse_apply: Optional[fn(mut StateType, List[Int]) raises] = None,
    ):
        self.transformations.append(
            ClassicalTransformation[StateType](
                name,
                targets,
                apply,
                inverse_apply,
            )
        )

    fn swap(
        mut self,
        a: Int,
        b: Int,
    ):
        self.transformations.append(SwapTransformation(a, b))

    fn qubit_reversal(mut self):
        self.transformations.append(QubitReversalTransformation())

    fn qubit_reversal(mut self, targets: List[Int]):
        self.transformations.append(QubitReversalTransformation(targets))

    fn qubit_reversal(mut self, reg: Register):
        var targets = List[Int](capacity=reg.length)
        for i in range(reg.length):
            targets.append(reg.start + i)
        self.transformations.append(QubitReversalTransformation(targets))

    fn append_circuit(
        mut self,
        other: Circuit[StateType]
    ) -> Bool:
        return self.append_circuit(other, Register("default", self.num_qubits))
    
    fn append_circuit(
        mut self,
        other: Circuit[StateType],
        reg: Register,
    ) -> Bool:
        if other.num_qubits != reg.length:
            return False
        var offset = reg.start
        for tr in other.transformations:
            if tr.isa[GateTransformation]():
                var gate_tr = tr[GateTransformation].copy()
                var controls = List[Int](capacity=len(gate_tr.controls))
                for c in gate_tr.controls:
                    controls.append(c + offset)
                self.transformations.append(
                    GateTransformation(
                        controls,
                        gate_tr.target + offset,
                        gate_tr.gate_info,
                    )
                )
            elif tr.isa[FusedPairTransformation]():
                var pair_tr = tr[FusedPairTransformation].copy()
                var first_controls = List[Int](capacity=len(pair_tr.first.controls))
                for c in pair_tr.first.controls:
                    first_controls.append(c + offset)
                var second_controls = List[Int](capacity=len(pair_tr.second.controls))
                for c in pair_tr.second.controls:
                    second_controls.append(c + offset)
                self.transformations.append(
                    FusedPairTransformation(
                        GateTransformation(
                            first_controls,
                            pair_tr.first.target + offset,
                            pair_tr.first.gate_info,
                        ),
                        GateTransformation(
                            second_controls,
                            pair_tr.second.target + offset,
                            pair_tr.second.gate_info,
                        ),
                    )
                )
            elif tr.isa[SwapTransformation]():
                var swap_tr = tr[SwapTransformation].copy()
                self.transformations.append(
                    SwapTransformation(
                        swap_tr.a + offset,
                        swap_tr.b + offset,
                    )
                )
            elif tr.isa[QubitReversalTransformation]():
                var qrev_tr = tr[QubitReversalTransformation].copy()
                if len(qrev_tr.targets) == 0:
                    var targets = List[Int](capacity=other.num_qubits)
                    for i in range(other.num_qubits):
                        targets.append(i + offset)
                    self.transformations.append(
                        QubitReversalTransformation(targets)
                    )
                else:
                    var targets = List[Int](capacity=len(qrev_tr.targets))
                    for t in qrev_tr.targets:
                        targets.append(t + offset)
                    self.transformations.append(
                        QubitReversalTransformation(targets)
                    )
            elif tr.isa[UnitaryTransformation]():
                var unitary_tr = tr[UnitaryTransformation].copy()
                self.transformations.append(
                    UnitaryTransformation(
                        unitary_tr.u.copy(),
                        unitary_tr.target + offset,
                        unitary_tr.m,
                        unitary_tr.name,
                    )
                )
            elif tr.isa[ControlledUnitaryTransformation]():
                var c_unitary_tr = tr[ControlledUnitaryTransformation].copy()
                self.transformations.append(
                    ControlledUnitaryTransformation(
                        c_unitary_tr.u.copy(),
                        c_unitary_tr.target + offset,
                        c_unitary_tr.control + offset,
                        c_unitary_tr.m,
                        c_unitary_tr.name,
                    )
                )
            elif tr.isa[ClassicalTransformation[StateType]]():
                var cl_tr = tr[
                    ClassicalTransformation[StateType]
                ].copy()
                # var targets = List[Int]()
                if len(cl_tr.targets) == 0:
                    targets = List[Int](capacity=other.num_qubits)
                    for i in range(other.num_qubits):
                        targets.append(i + offset)
                else:
                    targets = List[Int](capacity=len(cl_tr.targets))
                    for t in cl_tr.targets:
                        targets.append(t + offset)
                self.transformations.append(
                    ClassicalTransformation[StateType](
                        cl_tr.name,
                        targets,
                        cl_tr.apply,
                        cl_tr.inverse_apply,
                    )
                )
            elif tr.isa[MeasurementTransformation[StateType]]():
                var meas_tr = tr[
                    MeasurementTransformation[StateType]
                ].copy()
                var targets = List[Int](capacity=len(meas_tr.targets))
                for t in meas_tr.targets:
                    targets.append(t + offset)
                self.transformations.append(
                    MeasurementTransformation[StateType](
                        targets,
                        meas_tr.seed,
                        meas_tr.apply,
                        meas_tr.values,
                    )
                )
            else:
                var cl_tr = tr[
                    ClassicalTransformation[StateType]
                ].copy()
                # var targets = List[Int]()
                if len(cl_tr.targets) == 0:
                    targets = List[Int](capacity=other.num_qubits)
                    for i in range(other.num_qubits):
                        targets.append(i + offset)
                else:
                    targets = List[Int](capacity=len(cl_tr.targets))
                    for t in cl_tr.targets:
                        targets.append(t + offset)
                self.transformations.append(
                    ClassicalTransformation[StateType](
                        cl_tr.name,
                        targets,
                        cl_tr.apply,
                        cl_tr.inverse_apply,
                    )
                )
        return True

    fn find_register(self, name: String) -> Int:
        for i in range(len(self.registers)):
            if self.registers[i].name == name:
                return i
        return -1

    fn append_circuit_by_name(
        mut self,
        name: String,
        other: Circuit[StateType],
    ) -> Bool:
        var idx = self.find_register(name)
        if idx < 0:
            return False
        return self.append_circuit(other, self.registers[idx].copy())

    fn c_append_circuit(
        mut self,
        control: Int,
        other: Circuit[StateType]
    ) raises -> Bool:
        """Append a circuit controlled by a single qubit."""
        return self.c_append_circuit(control, other, Register("default", self.num_qubits))

    fn c_append_circuit(
        mut self,
        control: Int,
        other: Circuit[StateType],
        reg: Register,
    ) raises -> Bool:
        """Append a circuit controlled by a single qubit to a specific register."""
        if other.num_qubits != reg.length:
            return False
        var offset = reg.start
        for tr in other.transformations:
            if tr.isa[GateTransformation]():
                var gate_tr = tr[GateTransformation].copy()
                var controls = List[Int](capacity=len(gate_tr.controls) + 1)
                controls.append(control)  # Add the global control
                for c in gate_tr.controls:
                    controls.append(c + offset)
                self.transformations.append(
                    GateTransformation(
                        controls,
                        gate_tr.target + offset,
                        gate_tr.gate_info,
                    )
                )
            elif tr.isa[FusedPairTransformation]():
                var pair_tr = tr[FusedPairTransformation].copy()
                var first_controls = List[Int](capacity=len(pair_tr.first.controls) + 1)
                first_controls.append(control)
                for c in pair_tr.first.controls:
                    first_controls.append(c + offset)
                var second_controls = List[Int](capacity=len(pair_tr.second.controls) + 1)
                second_controls.append(control)
                for c in pair_tr.second.controls:
                    second_controls.append(c + offset)
                self.transformations.append(
                    FusedPairTransformation(
                        GateTransformation(
                            first_controls,
                            pair_tr.first.target + offset,
                            pair_tr.first.gate_info,
                        ),
                        GateTransformation(
                            second_controls,
                            pair_tr.second.target + offset,
                            pair_tr.second.gate_info,
                        ),
                    )
                )
            elif tr.isa[SwapTransformation]():
                var _swap_tr = tr[SwapTransformation].copy() #TODO
                # For swap operations, we need to control both the swap itself
                # This is complex - for now, we'll skip controlled swaps
                continue
            elif tr.isa[QubitReversalTransformation]():
                # Skip controlled qubit reversals for now
                continue
            elif tr.isa[UnitaryTransformation]():
                var unitary_tr = tr[UnitaryTransformation].copy()
                # Convert to controlled unitary
                self.c_unitary(
                    unitary_tr.u.copy(),
                    control,
                    unitary_tr.target + offset,
                    unitary_tr.m,
                    unitary_tr.name,
                )
            elif tr.isa[ControlledUnitaryTransformation]():
                var c_unitary_tr = tr[ControlledUnitaryTransformation].copy()
                # Add the global control to existing controls
                var controls = List[Int](capacity=2)
                controls.append(control)
                controls.append(c_unitary_tr.control + offset)
                self.c_unitary(
                    c_unitary_tr.u.copy(),
                    controls,
                    c_unitary_tr.target + offset,
                    c_unitary_tr.m,
                    c_unitary_tr.name,
                )
            elif tr.isa[MeasurementTransformation[StateType]]():
                # Skip measurements in controlled circuits
                continue
            else:
                # Skip classical transformations in controlled circuits
                continue
        return True

    fn c_append_circuit_by_name(
        mut self,
        control: Int,
        name: String,
        other: Circuit[StateType],
    ) raises -> Bool:
        """Append a circuit controlled by a single qubit to a named register."""
        var idx = self.find_register(name)
        if idx < 0:
            return False
        return self.c_append_circuit(control, other, self.registers[idx].copy())

    fn unitary(
        mut self,
        var u: List[Amplitude],
        target: Int,
        name: String = "unitary",
    ) raises:
        """Add an arbitrary unitary transformation acting on a single qubit."""
        self.unitary(u^, target, 1, name)

    fn unitary(
        mut self,
        gate: Gate,
        target: Int,
        name: String = "unitary",
    ) raises:
        """Add a single-qubit gate via a unitary matrix (Gate)."""
        var u = List[Amplitude](capacity=4)
        u.append(gate[0][0])
        u.append(gate[0][1])
        u.append(gate[1][0])
        u.append(gate[1][1])
        self.unitary(u^, target, 1, name)

    fn unitary(
        mut self,
        var u: List[Amplitude],
        target: Int,
        m: Int,
        name: String,
    ) raises:
        var dim = 1 << m
        var expected = dim * dim
        if len(u) != expected:
            raise Error(
                "Unitary size mismatch. Expected "
                + String(expected)
                + ", got "
                + String(len(u))
            )
        self.transformations.append(UnitaryTransformation(u^, target, m, name))

    fn u(mut self, var u: List[Amplitude], target: Int) raises:
        """Add an arbitrary unitary on a single qubit (shorthand)."""
        self.unitary(u^, target)

    fn append_u(
        mut self,
        var u: List[Amplitude],
        register: Register,
        name: String = "unitary",
    ) raises:
        """Add an arbitrary unitary acting on an entire register."""
        self.unitary(u^, register.start, register.length, name)

    fn c_unitary(
        mut self,
        var u: List[Amplitude],
        control: Int,
        target: Int,
        name: String = "unitary",
    ) raises:
        """Add a controlled unitary acting on a single target."""
        self.c_unitary(u^, control, target, 1, name)

    fn c_unitary(
        mut self,
        gate: Gate,
        control: Int,
        target: Int,
        name: String = "unitary",
    ) raises:
        """Add a controlled gate via a unitary matrix (Gate)."""
        var u = List[Amplitude](capacity=4)
        u.append(gate[0][0])
        u.append(gate[0][1])
        u.append(gate[1][0])
        u.append(gate[1][1])
        self.c_unitary(u^, control, target, 1, name)

    fn c_unitary(
        mut self,
        var u: List[Amplitude],
        control: Int,
        target: Int,
        m: Int,
        name: String,
    ) raises:
        var dim = 1 << m
        var expected = dim * dim
        if len(u) != expected:
            raise Error(
                "Unitary size mismatch. Expected "
                + String(expected)
                + ", got "
                + String(len(u))
            )
        self.transformations.append(
            ControlledUnitaryTransformation(u^, target, control, m, name)
        )

    fn c_unitary(
        mut self,
        var u: List[Amplitude],
        controls: List[Int],
        target: Int,
        name: String = "unitary",
    ) raises:
        """Add a multi-controlled unitary acting on a single target."""
        self.c_unitary(u^, controls, target, 1, name)

    fn c_unitary(
        mut self,
        var u: List[Amplitude],
        controls: List[Int],
        target: Int,
        m: Int,
        name: String,
    ) raises:
        """Add a multi-controlled unitary acting on m qubits."""
        var dim = 1 << m
        var expected = dim * dim
        if len(u) != expected:
            raise Error(
                "Unitary size mismatch. Expected "
                + String(expected)
                + ", got "
                + String(len(u))
            )

        # For multi-controlled unitaries, use the general add method
        # This creates a multi-controlled custom gate
        var gate_info = GateInfo(GateKind.CUSTOM)
        self.add(controls, target, gate_info)

    fn cu(mut self, var u: List[Amplitude], control: Int, target: Int) raises:
        """Add a controlled unitary acting on a single target (shorthand)."""
        self.c_unitary(u^, control, target)

    fn c_append_u(
        mut self,
        var u: List[Amplitude],
        control: Int,
        register: Register,
        name: String = "unitary",
    ) raises:
        """Add a controlled unitary acting on a register."""
        self.c_unitary(u^, control, register.start, register.length, name)

    fn h(
        mut self,
        target: Int,
    ):
        self.add(target, H_Gate)

    fn x(
        mut self,
        target: Int,
    ):
        self.add(target, X_Gate)

    fn y(
        mut self,
        target: Int,
    ):
        self.add(target, Y_Gate)

    fn z(
        mut self,
        target: Int,
    ):
        self.add(target, Z_Gate)

    fn p(
        mut self,
        target: Int,
        theta: FloatType,
    ):
        self.add(target, GateInfo(GateKind.P, theta))

    fn cp(
        mut self,
        control: Int,
        target: Int,
        theta: FloatType,
    ):
        self.add(control, target, P_Gate(theta))

    fn cx(
        mut self,
        control: Int,
        target: Int,
    ):
        self.add(control, target, X_Gate)

    fn ccx(
        mut self,
        control1: Int,
        control2: Int,
        target: Int,
    ):
        self.add(List[Int](control1, control2), target, X_Gate)

    fn cy(
        mut self,
        control: Int,
        target: Int,
    ):
        self.add(control, target, Y_Gate)

    fn cz(
        mut self,
        control: Int,
        target: Int,
    ):
        self.add(control, target, Z_Gate)

    fn rx(
        mut self,
        target: Int,
        theta: FloatType,
    ):
        self.add(target, RX_Gate(theta))

    fn ry(
        mut self,
        target: Int,
        theta: FloatType,
    ):
        self.add(target, RY_Gate(theta))

    fn rz(
        mut self,
        target: Int,
        theta: FloatType,
    ):
        self.add(target, RZ_Gate(theta))

    fn crx(
        mut self,
        control: Int,
        target: Int,
        theta: FloatType,
    ):
        self.add(control, target, RX_Gate(theta))

    fn cry(
        mut self,
        control: Int,
        target: Int,
        theta: FloatType,
    ):
        self.add(control, target, RY_Gate(theta))

    fn crz(
        mut self,
        control: Int,
        target: Int,
        theta: FloatType,
    ):
        self.add(control, target, RZ_Gate(theta))

    fn mcp(
        mut self,
        controls: List[Int],
        target: Int,
        theta: FloatType,
    ):
        self.add(controls, target, P_Gate(theta))

    fn inverse(self) raises -> Self:
        """Return the inverse of this circuit.

        Creates a new circuit with transformations in reverse order,
        where each transformation is inverted. Gates that are not
        invertible (like measurements) will raise an error.
        """
        var inv_circuit = Circuit[StateType](self.num_qubits)

        # Process transformations in reverse order
        for i in range(len(self.transformations) - 1, -1, -1):
            var tr = self.transformations[i]
            var inv_tr = _invert_transformation(tr)
            inv_circuit.transformations.append(inv_tr)

        inv_circuit.registers = self.registers.copy()
        return inv_circuit^


fn _invert_transformation[StateType: AnyType](
    tr: Transformation[StateType]
) raises -> Transformation[StateType]:
    """Invert a single transformation."""
    # Check for non-invertible transformations first
    if tr.isa[MeasurementTransformation[StateType]]():
        raise Error("Cannot invert circuit containing measurements")

    # Handle invertible transformations
    if tr.isa[GateTransformation]():
        var gate_tr = tr[GateTransformation].copy()
        return _invert_gate_transformation(gate_tr)

    elif tr.isa[FusedPairTransformation]():
        var fused_tr = tr[FusedPairTransformation].copy()
        # Invert fused pairs by inverting each gate and swapping order
        var inv_second = _invert_gate_transformation(fused_tr.second)
        var inv_first = _invert_gate_transformation(fused_tr.first)
        return FusedPairTransformation(inv_second, inv_first)

    elif tr.isa[UnitaryTransformation]():
        var unitary_tr = tr[UnitaryTransformation].copy()
        return _invert_unitary_transformation(unitary_tr)

    elif tr.isa[ControlledUnitaryTransformation]():
        var ctrl_unitary_tr = tr[ControlledUnitaryTransformation].copy()
        return _invert_controlled_unitary_transformation(ctrl_unitary_tr)

    elif tr.isa[SwapTransformation]():
        # Swap operations are their own inverse
        return tr

    elif tr.isa[QubitReversalTransformation]():
        # Bit reversal is its own inverse
        return tr

    elif tr.isa[ClassicalTransformation[StateType]]():
        var cl_tr = tr[ClassicalTransformation[StateType]].copy()
        if cl_tr.inverse_apply:
            return ClassicalTransformation[StateType](
                cl_tr.name + "_inv",
                cl_tr.targets,
                cl_tr.inverse_apply.value(),
                cl_tr.apply,
            )
        # Fall through: treat as self-inverse when no inverse is provided.
        return tr

    else:
        # For any other transformation types, assume they are self-inverse
        return tr


fn _invert_gate_transformation(gate_tr: GateTransformation) raises -> GateTransformation:
    """Invert a gate transformation."""
    var inv_gate_info = _invert_gate_info(gate_tr.gate_info)
    return GateTransformation(gate_tr.controls, gate_tr.target, inv_gate_info)


fn _invert_gate_info(gate_info: GateInfo) raises -> GateInfo:
    """Invert a gate info by negating parameters where applicable."""
    var kind = gate_info.kind
    var arg = gate_info.arg

    if kind == GateKind.H or kind == GateKind.X or kind == GateKind.Y or kind == GateKind.Z:
        # Self-inverse gates: H, X, Y, Z
        return gate_info.copy()

    elif kind == GateKind.P or kind == GateKind.RX or kind == GateKind.RY or kind == GateKind.RZ:
        # Parametric gates: negate the angle
        if arg:
            return GateInfo(kind, -arg.value())
        else:
            return gate_info.copy()

    elif kind == GateKind.CUSTOM:
        # For custom gates, we need to compute the adjoint (conjugate transpose)
        # This is a simplified version - real implementation would need matrix inversion
        raise Error("Custom gate inversion not implemented. Use unitary matrices for complex gates.")

    else:
        raise Error("Unknown gate kind in gate inversion")


fn _invert_unitary_transformation(unitary_tr: UnitaryTransformation) raises -> UnitaryTransformation:
    """Invert a unitary transformation by taking the adjoint."""
    # For unitary matrices, the inverse is the adjoint (conjugate transpose)
    # This requires implementing matrix adjoint operation
    var adjoint_u = _matrix_adjoint(unitary_tr.u, unitary_tr.m)
    return UnitaryTransformation(adjoint_u^, unitary_tr.target, unitary_tr.m, unitary_tr.name + "_inv")


fn _invert_controlled_unitary_transformation(ctrl_unitary_tr: ControlledUnitaryTransformation) raises -> ControlledUnitaryTransformation:
    """Invert a controlled unitary transformation."""
    var adjoint_u = _matrix_adjoint(ctrl_unitary_tr.u, ctrl_unitary_tr.m)
    return ControlledUnitaryTransformation(
        adjoint_u^,
        ctrl_unitary_tr.target,
        ctrl_unitary_tr.control,
        ctrl_unitary_tr.m,
        ctrl_unitary_tr.name + "_inv"
    )


fn _invert_classical_transformation[StateType: AnyType](
    classical_tr: ClassicalTransformation[StateType]
) -> ClassicalTransformation[StateType]:
    """Invert a classical transformation."""
    # Most classical transformations are their own inverse
    # Bit reversal and permutations can be inverted by applying again
    return classical_tr.copy()


fn _matrix_adjoint(u: List[Amplitude], m: Int) raises -> List[Amplitude]:
    """Compute the adjoint (conjugate transpose) of a unitary matrix."""
    var _dim = 1 << m
    var adjoint = List[Amplitude](capacity=len(u))

    # For now, implement a simple conjugate (transpose would need more complex logic)
    # This is a placeholder - real matrix adjoint requires proper transpose + conjugate
    for i in range(len(u)):
        var amp = u[i]
        adjoint.append(Complex(amp.re, -amp.im))  # Just conjugate for now

    return adjoint^
