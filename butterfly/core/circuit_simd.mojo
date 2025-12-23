"""
Parameterized QuantumCircuit for compile-time SIMD optimization.

This version reuses QuantumCircuit for gate building and transformation
management, while providing SIMD-optimized execution with compile-time N.
"""

from butterfly.core.state import (
    QuantumState,
    transform_simd,
    c_transform_simd,
    bit_reverse_state,
    mc_transform_interval,
    c_transform_simd_base_v2,
)
from butterfly.core.types import *
from butterfly.core.gates import *
from butterfly.core.circuit import (
    QuantumCircuit,
    Transformation,
    GateTransformation,
    SingleControlGateTransformation,
    MultiControlGateTransformation,
    BitReversalTransformation,
)
from butterfly.core.c_transform_fast_v2 import (
    c_transform_h_simd_v2,
    c_transform_x_simd_v2,
    c_transform_p_simd_v2,
)


struct QuantumCircuitSIMD[n: Int](Copyable):
    """
    Compile-time parameterized quantum circuit for SIMD optimization.

    This version delegates gate building to QuantumCircuit while providing
    SIMD-optimized execution that leverages compile-time knowledge of N.

    Parameters:
        n: Number of qubits (compile-time constant)
    """

    var circuit: QuantumCircuit  # Reuse for gate building
    var state: QuantumState

    fn __init__(out self):
        """Initialize circuit with n qubits in |0⟩ state."""
        self.circuit = QuantumCircuit(n)
        self.state = QuantumState(n)

    fn __copyinit__(out self, existing: Self):
        self.circuit = existing.circuit.copy()
        self.state = existing.state.copy()

    # Delegate all gate methods to QuantumCircuit
    fn h(mut self, target: Int):
        """Apply Hadamard gate."""
        self.circuit.h(target)

    fn x(mut self, target: Int):
        """Apply Pauli-X gate."""
        self.circuit.x(target)

    fn y(mut self, target: Int):
        """Apply Pauli-Y gate."""
        self.circuit.y(target)

    fn z(mut self, target: Int):
        """Apply Pauli-Z gate."""
        self.circuit.z(target)

    fn cx(mut self, control: Int, target: Int):
        """Apply CNOT gate."""
        self.circuit.cx(control, target)

    fn cz(mut self, control: Int, target: Int):
        """Apply controlled-Z gate."""
        self.circuit.cz(control, target)

    fn rx(mut self, target: Int, theta: FloatType):
        """Apply RX rotation."""
        self.circuit.rx(target, theta)

    fn ry(mut self, target: Int, theta: FloatType):
        """Apply RY rotation."""
        self.circuit.ry(target, theta)

    fn rz(mut self, target: Int, theta: FloatType):
        """Apply RZ rotation."""
        self.circuit.rz(target, theta)

    fn p(mut self, target: Int, theta: FloatType):
        """Apply phase gate."""
        self.circuit.p(target, theta)

    fn add(mut self, gate: Gate, target: Int):
        """Add a single-qubit gate (low-level)."""
        self.circuit.add(gate, target)

    fn add_controlled(mut self, gate: Gate, target: Int, control: Int):
        """Add a controlled gate (low-level)."""
        self.circuit.add_controlled(gate, target, control)

    fn bit_reverse(mut self):
        """Add bit-reversal permutation."""
        self.circuit.bit_reverse()

    fn execute_simd(mut self):
        """
        Execute circuit using SIMD transforms with compile-time N.
        No dispatch overhead - directly uses transform_simd[N].
        """
        alias N = 1 << n

        for i in range(len(self.circuit.transformations)):
            var t = self.circuit.transformations[i]

            if t.isa[GateTransformation]():
                var g = t[GateTransformation].copy()
                transform_simd[N](self.state, g.target, g.gate)
            elif t.isa[SingleControlGateTransformation]():
                var g = t[SingleControlGateTransformation].copy()
                c_transform_simd[N](self.state, g.control, g.target, g.gate)
            elif t.isa[MultiControlGateTransformation]():
                var g = t[MultiControlGateTransformation].copy()
                mc_transform_interval(self.state, g.controls, g.target, g.gate)
            elif t.isa[BitReversalTransformation]():
                bit_reverse_state(self.state)

    fn execute_simd_v2(mut self):
        """
        Execute circuit using SIMD transforms v2 with compile-time N.
        Uses specialized kernels and optimized indexing.
        """
        alias N = 1 << n
        from butterfly.core.execute_simd_v2_dispatch import (
            execute_transformations_simd_v2,
        )

        execute_transformations_simd_v2[N](
            self.state, self.circuit.transformations
        )

    fn execute_fused_v3(mut self):
        """
        Execute circuit using v3 fusion with compile-time N.
        Fuses disjoint gates into Kronecker products for maximum performance.
        """
        alias N = 1 << n
        from butterfly.core.execute_fused_v3 import execute_fused_v3

        execute_fused_v3[N](self.state, self.circuit.transformations)

    fn run(mut self):
        """
        Direct, no-dispatch execution of SIMD v2 transformations.
        Bypasses intermediate helper modules and runtime dispatch logic.
        """
        alias N = 1 << n

        for i in range(len(self.circuit.transformations)):
            var t = self.circuit.transformations[i]

            if t.isa[GateTransformation]():
                var g = t[GateTransformation].copy()
                transform_simd[N](self.state, g.target, g.gate)
            elif t.isa[BitReversalTransformation]():
                bit_reverse_state(self.state)
            elif t.isa[MultiControlGateTransformation]():
                var g = t[MultiControlGateTransformation].copy()
                mc_transform_interval(self.state, g.controls, g.target, g.gate)
            elif t.isa[SingleControlGateTransformation]():
                var g = t[SingleControlGateTransformation].copy()
                # Specialized kernels v2
                if is_h(g.gate):
                    c_transform_h_simd_v2(self.state, g.control, g.target)
                elif is_x(g.gate):
                    c_transform_x_simd_v2(self.state, g.control, g.target)
                elif is_p(g.gate):
                    var theta = get_phase_angle(g.gate)
                    c_transform_p_simd_v2(
                        self.state, g.control, g.target, theta
                    )
                else:
                    var stride = 1 << g.target
                    c_transform_simd_base_v2[N](
                        self.state, g.control, stride, g.gate
                    )

    fn num_transformations(self) -> Int:
        """Return the number of transformations in the circuit."""
        return len(self.circuit.transformations)
