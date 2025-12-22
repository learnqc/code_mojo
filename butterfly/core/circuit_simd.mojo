"""
Parameterized QuantumCircuit for compile-time SIMD optimization.

This is an alternative to the runtime QuantumCircuit that enables
direct use of transform_simd[N] without dispatch overhead.
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

    Unlike QuantumCircuit, this version knows the number of qubits at
    compile-time, enabling direct use of transform_simd[N] without dispatch.

    Parameters:
        n: Number of qubits (compile-time constant)
    """

    var state: QuantumState
    var transformations: List[Transformation]

    fn __init__(out self):
        """Initialize circuit with n qubits in |0⟩ state."""
        self.state = QuantumState(n)
        self.transformations = List[Transformation]()

    fn __copyinit__(out self, existing: Self):
        self.state = existing.state.copy()
        self.transformations = List[Transformation](
            capacity=len(existing.transformations)
        )
        for i in range(len(existing.transformations)):
            self.transformations.append(existing.transformations[i])

    fn add(mut self, gate: Gate, target: Int):
        """Add a single-qubit gate to the circuit."""
        self.transformations.append(GateTransformation(gate, target))

    fn add_controlled(mut self, gate: Gate, target: Int, control: Int):
        """Add a controlled gate to the circuit."""
        self.transformations.append(
            SingleControlGateTransformation(gate, target, control)
        )

    fn bit_reverse(mut self):
        """Add a bit-reversal permutation to the circuit."""
        self.transformations.append(BitReversalTransformation())

    fn execute_simd(mut self):
        """
        Execute circuit using SIMD transforms with compile-time N.
        No dispatch overhead - directly uses transform_simd[N].
        """
        alias N = 1 << n

        for i in range(len(self.transformations)):
        for i in range(len(self.transformations)):
            var t = self.transformations[i]

            if t.isa[GateTransformation]():
                var g = t[GateTransformation]
                transform_simd[N](self.state, g.target, g.gate)
            elif t.isa[SingleControlGateTransformation]():
                var g = t[SingleControlGateTransformation]
                c_transform_simd[N](
                    self.state, g.control, g.target, g.gate
                )
            elif t.isa[MultiControlGateTransformation]():
                var g = t[MultiControlGateTransformation]
                mc_transform_interval(
                    self.state, g.controls, g.target, g.gate
                )
            elif t.isa[BitReversalTransformation]():
                bit_reverse_state(self.state)

    fn execute_simd_v2(mut self):
        """
        Execute circuit using SIMD transforms v2 with compile-time N.
        Uses specialized kernels and optimized indexing.
        Dispatches to helper module.
        """
        alias N = 1 << n
        from butterfly.core.execute_simd_v2_dispatch import (
            execute_transformations_simd_v2,
        )

        execute_transformations_simd_v2[N](self.state, self.transformations)

    fn run(mut self):
        """
        Direct, no-dispatch execution of SIMD v2 transformations.
        Bypasses intermediate helper modules and runtime dispatch logic.
        """
        alias N = 1 << n

        for i in range(len(self.transformations)):
        for i in range(len(self.transformations)):
            var t = self.transformations[i]

            if t.isa[GateTransformation]():
                var g = t[GateTransformation]
                transform_simd[N](self.state, g.target, g.gate)
            elif t.isa[BitReversalTransformation]():
                bit_reverse_state(self.state)
            elif t.isa[MultiControlGateTransformation]():
                var g = t[MultiControlGateTransformation]
                mc_transform_interval(
                    self.state, g.controls, g.target, g.gate
                )
            elif t.isa[SingleControlGateTransformation]():
                var g = t[SingleControlGateTransformation]
                # Specialized kernels v2
                if is_h(g.gate):
                    c_transform_h_simd_v2(
                        self.state, g.control, g.target
                    )
                elif is_x(g.gate):
                    c_transform_x_simd_v2(
                        self.state, g.control, g.target
                    )
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
        return len(self.transformations)
