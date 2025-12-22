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
from butterfly.core.circuit import QuantumTransformation


fn execute_transformations_simd[
    N: Int
](mut state: QuantumState, transformations: List[QuantumTransformation]):
    """Execute transformations using SIMD with compile-time N."""
    for i in range(len(transformations)):
        var t = transformations[i].copy()

        if t.is_permutation:
            bit_reverse_state(state)
            continue

        if t.is_controlled():
            if t.num_controls() == 1:
                c_transform_simd[N](state, t.controls[0], t.target, t.gate)
            else:
                mc_transform_interval(state, t.controls, t.target, t.gate)
        else:
            transform_simd[N](state, t.target, t.gate)
