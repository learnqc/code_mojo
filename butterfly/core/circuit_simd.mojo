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
from butterfly.core.circuit import QuantumTransformation
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
    var transformations: List[QuantumTransformation]

    fn __init__(out self):
        """Initialize circuit with n qubits in |0⟩ state."""
        self.state = QuantumState(n)
        self.transformations = List[QuantumTransformation]()

    fn __copyinit__(out self, existing: Self):
        self.state = existing.state.copy()
        self.transformations = List[QuantumTransformation](
            capacity=len(existing.transformations)
        )
        for i in range(len(existing.transformations)):
            self.transformations.append(existing.transformations[i].copy())

    fn add(mut self, gate: Gate, target: Int):
        """Add a single-qubit gate to the circuit."""
        self.transformations.append(QuantumTransformation(gate, target))

    fn add_controlled(mut self, gate: Gate, target: Int, control: Int):
        """Add a controlled gate to the circuit."""
        var controls = List[Int]()
        controls.append(control)
        self.transformations.append(
            QuantumTransformation(gate, target, controls^)
        )

    fn bit_reverse(mut self):
        """Add a bit-reversal permutation to the circuit."""
        var t = QuantumTransformation(is_permutation=True)
        self.transformations.append(t^)

    fn execute_simd(mut self):
        """
        Execute circuit using SIMD transforms with compile-time N.
        No dispatch overhead - directly uses transform_simd[N].
        """
        alias N = 1 << n

        for i in range(len(self.transformations)):
            var t = self.transformations[i].copy()

            if t.is_permutation:
                bit_reverse_state(self.state)
            elif t.is_controlled():
                if t.num_controls() == 1:
                    c_transform_simd[N](
                        self.state, t.controls[0], t.target, t.gate
                    )
                else:
                    mc_transform_interval(
                        self.state, t.controls, t.target, t.gate
                    )
            else:
                transform_simd[N](self.state, t.target, t.gate)

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
            var t = self.transformations[i].copy()

            if t.is_permutation:
                bit_reverse_state(self.state)
            elif t.is_controlled():
                if t.num_controls() == 1:
                    # Specialized kernels v2
                    if is_h(t.gate):
                        c_transform_h_simd_v2(
                            self.state, t.controls[0], t.target
                        )
                    elif is_x(t.gate):
                        c_transform_x_simd_v2(
                            self.state, t.controls[0], t.target
                        )
                    elif is_p(t.gate):
                        var theta = get_phase_angle(t.gate)
                        c_transform_p_simd_v2(
                            self.state, t.controls[0], t.target, theta
                        )
                    else:
                        var stride = 1 << t.target
                        c_transform_simd_base_v2[N](
                            self.state, t.controls[0], stride, t.gate
                        )
                else:
                    mc_transform_interval(
                        self.state, t.controls, t.target, t.gate
                    )
            else:
                transform_simd[N](self.state, t.target, t.gate)

    fn num_transformations(self) -> Int:
        """Return the number of transformations in the circuit."""
        return len(self.transformations)
