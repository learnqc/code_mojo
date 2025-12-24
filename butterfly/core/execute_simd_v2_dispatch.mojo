"""
Helper module for execute_simd_v2 with compile-time N dispatch.
"""
from butterfly.core.state import (
    QuantumState,
    transform_simd,
    bit_reverse_state,
    mc_transform_interval,
    c_transform_simd_base_v2,
)
from butterfly.core.gates import is_h, is_x, is_p, get_phase_angle
from butterfly.core.types import Gate
from butterfly.core.circuit import (
    Transformation,
    GateTransformation,
    SingleControlGateTransformation,
    MultiControlGateTransformation,
    BitReversalTransformation,
    DiagonalTransformation,
)
from butterfly.core.c_transform_fast_v2 import (
    c_transform_h_simd_v2,
    c_transform_x_simd_v2,
    c_transform_p_simd_v2,
)


fn execute_transformations_simd_v2[
    N: Int
](mut state: QuantumState, transformations: List[Transformation]):
    """Execute transformations using SIMD v2 with compile-time N."""
    for i in range(len(transformations)):
        var t = transformations[i]

        if t.isa[BitReversalTransformation]():
            bit_reverse_state(state)
        elif t.isa[SingleControlGateTransformation]():
            var g = t[SingleControlGateTransformation].copy()
            # Specialized kernels v2
            if is_h(g.gate):
                c_transform_h_simd_v2(state, g.control, g.target)
            elif is_x(g.gate):
                c_transform_x_simd_v2(state, g.control, g.target)
            elif is_p(g.gate):
                var theta = get_phase_angle(g.gate)
                c_transform_p_simd_v2(state, g.control, g.target, theta)
            else:
                var stride = 1 << g.target
                c_transform_simd_base_v2[N](state, g.control, stride, g.gate)
        elif t.isa[MultiControlGateTransformation]():
            var g = t[MultiControlGateTransformation].copy()
            mc_transform_interval(state, g.controls, g.target, g.gate)
        elif t.isa[GateTransformation]():
            var g = t[GateTransformation].copy()
            transform_simd[N](state, g.target, g.gate)
        elif t.isa[DiagonalTransformation]():
            # Handle diagonal phase flip transformation
            from butterfly.core.types import Amplitude

            var g = t[DiagonalTransformation].copy()
            var n_qubits = 0
            var temp = N
            while temp > 1:
                temp >>= 1
                n_qubits += 1
            var n_shortcut = g.size if g.size > 0 else n_qubits
            var mask = (1 << n_shortcut) - 1
            for k in range(N):
                var val = (k >> g.offset) & mask
                # Check if val is in items
                for p_idx in range(len(g.items)):
                    if val == g.items[p_idx]:
                        state[k] = Amplitude(-state[k].re, -state[k].im)
                        break
