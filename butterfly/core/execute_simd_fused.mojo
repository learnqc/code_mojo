"""
execute_simd_fused: Clean fusion strategy combining SIMD v2 with gate fusion.

Uses proven patterns from v_grid_fused without the complexity of grid partitioning.
"""

from butterfly.core.state import QuantumState
from butterfly.core.circuit import (
    QuantumCircuit,
    Transformation,
    is_permutation,
    is_controlled,
    get_target,
    get_gate,
)
from butterfly.algos.fused_gates import compute_kron_product, transform_fused
from butterfly.algos.unitary_kernels import Matrix4x4


struct FusionGroup(Copyable, Movable):
    """A group of transformations that can be executed together."""

    var transformations: List[Transformation]
    var is_fused: Bool

    fn __init__(out self, is_fused: Bool = False):
        self.transformations = List[Transformation]()
        self.is_fused = is_fused

    fn __copyinit__(out self, existing: Self):
        self.transformations = List[Transformation]()
        for i in range(len(existing.transformations)):
            self.transformations.append(existing.transformations[i])
        self.is_fused = existing.is_fused

    fn __moveinit__(out self, deinit existing: Self):
        self.transformations = existing.transformations^
        self.is_fused = existing.is_fused

    fn copy(self) -> Self:
        var new_g = FusionGroup(self.is_fused)
        for i in range(len(self.transformations)):
            new_g.transformations.append(self.transformations[i])
        return new_g^


fn can_fuse(t1: Transformation, t2: Transformation) -> Bool:
    """Check if two transformations can be fused into a 4x4 matrix."""
    # Don't fuse controlled gates (use specialized kernels instead)
    if is_controlled(t1) or is_controlled(t2):
        return False

    # Don't fuse permutations
    if is_permutation(t1) or is_permutation(t2):
        return False

    # Must target different qubits
    if get_target(t1) == get_target(t2):
        return False

    return True


fn analyze_for_fusion(
    transformations: List[Transformation],
) -> List[FusionGroup]:
    """Analyze circuit and group transformations for fusion."""
    var groups = List[FusionGroup]()
    var i = 0
    var n = len(transformations)

    while i < n:
        var t = transformations[i]

        # Try to fuse with next gate
        if i + 1 < n:
            var next_t = transformations[i + 1]
            if can_fuse(t, next_t):
                # Fuse these two gates
                var g = FusionGroup(is_fused=True)
                g.transformations.append(t)
                g.transformations.append(next_t)
                groups.append(g^)
                i += 2
                continue

        # Single gate (not fused)
        var g = FusionGroup(is_fused=False)
        g.transformations.append(t)
        groups.append(g^)
        i += 1

    return groups^


fn execute_fused_pair(
    mut state: QuantumState, transformations: List[Transformation]
):
    """Execute two gates fused into a single 4x4 matrix operation."""
    var t0 = transformations[0]
    var t1 = transformations[1]

    var q0 = get_target(t0)
    var q1 = get_target(t1)
    var g0 = get_gate(t0)
    var g1 = get_gate(t1)

    # Use proven transform_fused from v_grid_fused
    transform_fused(state, q0, g0, q1, g1)


fn execute_single_gate(mut state: QuantumState, t: Transformation):
    """Execute a single gate using SIMD v2 kernels."""
    if is_permutation(t):
        from butterfly.core.state import bit_reverse_state

        bit_reverse_state(state)
    elif is_controlled(t):
        # Use proven row-local controlled gate kernels from v_grid
        from butterfly.core.c_transform_fast_v2 import (
            c_transform_h_simd_v2,
            c_transform_p_simd_v2,
        )
        from butterfly.core.circuit import SingleControlGateTransformation
        from butterfly.core.gates import is_h, is_p, get_phase_angle

        var sct = t[SingleControlGateTransformation].copy()
        var control = sct.control
        var target = sct.target
        var gate = sct.gate

        if is_h(gate):
            c_transform_h_simd_v2(state, control, target)
        elif is_p(gate):
            c_transform_p_simd_v2(state, control, target, get_phase_angle(gate))
    else:
        # Uncontrolled gate - use SIMD v2
        from butterfly.core.state import transform

        transform(state, get_target(t), get_gate(t))


fn execute_simd_fused(mut state: QuantumState, circuit: QuantumCircuit):
    """Execute circuit using SIMD with gate fusion.

    Args:
        state: The quantum state to transform.
        circuit: The circuit to execute.
    """
    # Analyze circuit for fusible pairs
    var groups = analyze_for_fusion(circuit.transformations)

    # Execute each group
    for i in range(len(groups)):
        var g = groups[i].copy()

        if g.is_fused:
            execute_fused_pair(state, g.transformations)
        else:
            execute_single_gate(state, g.transformations[0])
