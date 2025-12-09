from butterfly.core.state import QuantumState, transform, c_transform, mc_transform_interval
from butterfly.core.types import *
from butterfly.core.gates import *


struct QuantumTransformation(Copyable, Movable):
    """A quantum transformation representing a gate applied to target qubit(s) with optional controls."""
    var gate: Gate
    var target: Int
    var controls: List[Int]

    fn __init__(out self, gate: Gate, target: Int):
        self.gate = gate
        self.target = target
        self.controls = List[Int]()

    fn __init__(out self, gate: Gate, target: Int, var controls: List[Int]):
        self.gate = gate
        self.target = target
        self.controls = controls^

    fn __copyinit__(out self, existing: Self):
        self.gate = existing.gate
        self.target = existing.target
        self.controls = List[Int](capacity=len(existing.controls))
        for q in existing.controls:
            self.controls.append(q)

    fn __moveinit__(out self, deinit existing: Self):
        self.gate = existing.gate
        self.target = existing.target
        self.controls = existing.controls^

    fn is_controlled(self) -> Bool:
        return len(self.controls) > 0

    fn num_controls(self) -> Int:
        return len(self.controls)

    fn add_control(mut self, q: Int):
        self.controls.append(q)


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


struct QuantumCircuit(ImplicitlyCopyable):
    """A quantum circuit that manages quantum state and transformation operations.

    The circuit maintains a quantum state and a sequence of transformations (gates)
    that can be applied to the state. Supports quantum registers for organizing qubits.
    """
    var state: QuantumState
    var transformations: List[QuantumTransformation]
    var num_qubits: Int
    var registers: List[QuantumRegister]

    fn __init__(out self, num_qubits: Int):
        """Initialize a quantum circuit with n qubits in the |0⟩ state."""
        self.num_qubits = num_qubits
        self.state = QuantumState(num_qubits)
        self.transformations = List[QuantumTransformation]()
        self.registers = List[QuantumRegister]()

    fn __init__(out self, var state: QuantumState, num_qubits: Int):
        """Initialize a quantum circuit with a given state."""
        self.num_qubits = num_qubits
        self.state = state^
        self.transformations = List[QuantumTransformation]()
        self.registers = List[QuantumRegister]()

    fn __copyinit__(out self, existing: Self):
        self.num_qubits = existing.num_qubits
        self.state = existing.state
        self.transformations = List[QuantumTransformation](capacity=len(existing.transformations))
        for i in range(len(existing.transformations)):
            self.transformations.append(existing.transformations[i].copy())
        self.registers = List[QuantumRegister](capacity=len(existing.registers))
        for i in range(len(existing.registers)):
            self.registers.append(existing.registers[i].copy())

    fn __moveinit__(out self, deinit existing: Self):
        self.num_qubits = existing.num_qubits
        self.state = existing.state^
        self.transformations = existing.transformations^
        self.registers = existing.registers^

    fn add(mut self, gate: Gate, target: Int):
        """Add a gate to the circuit on the specified target qubit."""
        var transformation = QuantumTransformation(gate, target)
        self.transformations.append(transformation^)

    fn add_controlled(mut self, gate: Gate, target: Int, control: Int):
        """Add a controlled gate to the circuit."""
        var controls = List[Int]()
        controls.append(control)
        var transformation = QuantumTransformation(gate, target, controls^)
        self.transformations.append(transformation^)

    fn add_multi_controlled(mut self, gate: Gate, target: Int, var controls: List[Int]):
        """Add a multi-controlled gate to the circuit."""
        var transformation = QuantumTransformation(gate, target, controls^)
        self.transformations.append(transformation^)

    fn apply_transformation(mut self, t: QuantumTransformation):
        """Apply a single transformation to the state."""
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

    fn clear_transformations(mut self):
        """Clear all transformations from the circuit."""
        self.transformations = List[QuantumTransformation]()

    fn num_transformations(self) -> Int:
        """Return the number of transformations in the circuit."""
        return len(self.transformations)

    fn get_state(self) -> QuantumState:
        """Return a copy of the current quantum state."""
        return self.state

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
