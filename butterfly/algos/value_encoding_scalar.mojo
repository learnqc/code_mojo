from butterfly.core.state import QuantumState
from butterfly.core.types import FloatType, Gate, Int, List, pi
from butterfly.core.gates import *
from butterfly.core.transformations_scalar import transform_scalar, c_transform_scalar
from butterfly.core.state import bit_reverse_state
from butterfly.utils.context import ExecContext

fn iqft_scalar(
    mut state: QuantumState,
    targets: List[Int],
    swap: Bool = False,
    ctx: ExecContext = ExecContext(),
):
    for j in reversed(range(len(targets))):
        transform_scalar(state, targets[j], H, GateKind.H, 0, ctx)
        for k in reversed(range(j)):
            c_transform_scalar(
                state,
                targets[j],
                targets[k],
                P(-pi / 2 ** (j - k)),
                GateKind.P,
                FloatType(-pi / 2 ** (j - k)),
                ctx,
            )

    if swap:
        bit_reverse_state(state)


fn prep_scalar(
    n: Int,
    v: FloatType,
    swap: Bool = False,
    ctx: ExecContext = ExecContext(),
) -> QuantumState:
    var state = QuantumState(n)
    for j in range(n):
        transform_scalar(state, j, H, GateKind.H, 0, ctx)
    for j in range(n):
        transform_scalar(
            state,
            j,
            P(2 * pi / 2 ** (j + 1) * v),
            GateKind.P,
            FloatType(2 * pi / 2 ** (j + 1) * v),
            ctx,
        )
    _ = swap
    return state^

fn encode_value_scalar(
    n: Int,
    v: FloatType,
    swap: Bool = False,
    ctx: ExecContext = ExecContext(),
) -> QuantumState:
    var state = QuantumState(n)

    for j in range(n):
        transform_scalar(state, j, H, GateKind.H, 0, ctx)

    for j in range(n):
        transform_scalar(
            state,
            j,
            P(2 * pi / 2 ** (j + 1) * v),
            GateKind.P,
            FloatType(2 * pi / 2 ** (j + 1) * v),
            ctx,
        )

    iqft_scalar(state, [n - 1 - j for j in range(n)], swap=swap, ctx=ctx)
    return state^
