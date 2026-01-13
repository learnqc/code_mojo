from butterfly.core.state import QuantumState
from butterfly.core.types import FloatType, Gate, Int, List, pi
from butterfly.core.gates import *
from butterfly.core.transformations_simd_parallel import (
    transform_gate_simd_parallel,
    transform_h_simd_parallel,
    transform_p_simd_parallel,
    c_transform_p_simd_parallel,
    mc_transform_simd_parallel,
)
from butterfly.core.state import bit_reverse_state
from butterfly.utils.context import ExecContext



fn prep_simd_parallel(
    n: Int,
    v: FloatType,
    ctx: ExecContext = ExecContext(),
) -> QuantumState:
    var state = QuantumState(n)
    for j in range(n):
        if ctx.simd_use_specialized_h:
            transform_h_simd_parallel(state, j, ctx)
        else:
            transform_gate_simd_parallel(state, j, H, ctx)
    for j in range(n):
        var theta = 2 * pi / 2 ** (j + 1) * v
        if ctx.simd_use_specialized_p:
            transform_p_simd_parallel(state, j, theta, ctx)
        else:
            transform_gate_simd_parallel(state, j, P(theta), ctx)
    return state^


fn iqft_simd_parallel(
    mut state: QuantumState,
    targets: List[Int],
    swap: Bool = False,
    ctx: ExecContext = ExecContext(),
):
    for j in reversed(range(len(targets))):
        if ctx.simd_use_specialized_h:
            transform_h_simd_parallel(state, targets[j], ctx)
        else:
            transform_gate_simd_parallel(state, targets[j], H, ctx)
        for k in reversed(range(j)):
            var theta = -pi / 2 ** (j - k)
            if ctx.simd_use_specialized_cp:
                c_transform_p_simd_parallel(
                    state,
                    targets[j],
                    targets[k],
                    theta,
                    ctx,
                )
            else:
                mc_transform_simd_parallel(
                    state,
                    List[Int](targets[j]),
                    targets[k],
                    P(theta),
                    ctx,
                )

    if swap:
        bit_reverse_state(state)


fn encode_value_simd_parallel(
    n: Int,
    v: FloatType,
    swap: Bool = False,
    ctx: ExecContext = ExecContext(),
) -> QuantumState:
    var state = QuantumState(n)

    for j in range(n):
        if ctx.simd_use_specialized_h:
            transform_h_simd_parallel(state, j, ctx)
        else:
            transform_gate_simd_parallel(state, j, H, ctx)

    for j in range(n):
        if swap:
            var theta = 2 * pi / 2 ** (n - j) * v
            if ctx.simd_use_specialized_p:
                transform_p_simd_parallel(state, j, theta, ctx)
            else:
                transform_gate_simd_parallel(state, j, P(theta), ctx)
        else:
            var theta = 2 * pi / 2 ** (j + 1) * v
            if ctx.simd_use_specialized_p:
                transform_p_simd_parallel(state, j, theta, ctx)
            else:
                transform_gate_simd_parallel(state, j, P(theta), ctx)

    var targets = List[Int]()
    for j in range(n):
        targets.append(n - 1 - j)
    if swap:
        targets = List[Int]()
        for j in range(n):
            targets.append(j)

    iqft_simd_parallel(state, targets, swap=swap, ctx=ctx)
    return state^
