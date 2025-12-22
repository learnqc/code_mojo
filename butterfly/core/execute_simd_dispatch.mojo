"""
Helper module for execute_simd with compile-time N dispatch.
"""
from butterfly.core.state import (
    QuantumState,
    transform_simd,
    c_transform_simd,
    bit_reverse_state,
    transform,
    c_transform,
    mc_transform_interval,
)
from butterfly.core.types import Gate
from butterfly.core.circuit import (
    Transformation,
    GateTransformation,
    SingleControlGateTransformation,
    MultiControlGateTransformation,
    BitReversalTransformation,
)


fn execute_transformations_simd[
    N: Int
](mut state: QuantumState, transformations: List[Transformation]):
    """Execute transformations using SIMD with compile-time N."""
    for i in range(len(transformations)):
        var t = transformations[i]

        if t.isa[BitReversalTransformation]():
            bit_reverse_state(state)
        elif t.isa[SingleControlGateTransformation]():
            var g = t[SingleControlGateTransformation].copy()
            c_transform_simd[N](state, g.control, g.target, g.gate)
        elif t.isa[MultiControlGateTransformation]():
            var g = t[MultiControlGateTransformation].copy()
            mc_transform_interval(state, g.controls, g.target, g.gate)
        elif t.isa[GateTransformation]():
            var g = t[GateTransformation].copy()
            transform_simd[N](state, g.target, g.gate)
