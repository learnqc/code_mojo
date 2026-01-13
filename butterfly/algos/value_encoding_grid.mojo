from butterfly.core.state import QuantumState, bit_reverse_state
from butterfly.core.types import FloatType
from butterfly.core.transformations_grid import (
    transform_h_grid,
    transform_p_grid,
    c_transform_p_grid,
)
from butterfly.utils.context import ExecContext
from collections import List
from math import pi


fn iqft_grid(
    mut state: QuantumState,
    col_bits: Int,
    targets: List[Int],
    swap: Bool = False,
    ctx: ExecContext = ExecContext(),
) raises:
    for j in reversed(range(len(targets))):
        transform_h_grid[8](state, col_bits, targets[j], ctx)
        for k in reversed(range(j)):
            c_transform_p_grid[8](
                state,
                col_bits,
                targets[j],
                targets[k],
                -pi / 2 ** (j - k),
                ctx,
            )

    if swap:
        bit_reverse_state(state)


fn prep_grid(
    n: Int,
    col_bits: Int,
    v: FloatType,
    swap: Bool = False,
    ctx: ExecContext = ExecContext(),
) raises -> QuantumState:
    var state = QuantumState(n)
    for j in range(n):
        transform_h_grid[8](state, col_bits, j, ctx)
    for j in range(n):
        transform_p_grid[8](
            state,
            col_bits,
            j,
            2 * pi / 2 ** (j + 1) * v,
            ctx,
        )
    _ = swap
    return state^


fn encode_value_grid(
    n: Int,
    col_bits: Int,
    v: FloatType,
    swap: Bool = False,
    ctx: ExecContext = ExecContext(),
) raises -> QuantumState:
    var state = QuantumState(n)

    for j in range(n):
        transform_h_grid[8](state, col_bits, j, ctx)

    for j in range(n):
        transform_p_grid[8](
            state,
            col_bits,
            j,
            2 * pi / 2 ** (j + 1) * v,
            ctx,
        )

    iqft_grid(state, col_bits, [n - 1 - j for j in range(n)], swap=swap, ctx=ctx)
    return state^
