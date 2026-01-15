from butterfly.core.state import QuantumState
from butterfly.core.types import FloatType, Gate, Int, List, pi
from butterfly.core.gates import *
from butterfly.core.transformations_simd import transform_simd, c_transform_simd, transform_h_simd, transform_p_simd, c_transform_p_simd
from butterfly.core.state import bit_reverse_state
from butterfly.utils.context import ExecContext

fn prep_simd(
    n: Int,
    v: FloatType,
    swap: Bool = False,
    ctx: ExecContext = ExecContext(),
) -> QuantumState:
    var state = QuantumState(n)

    for j in range(n):
        transform_simd(state, j, H, GateKind.H, ctx=ctx)

    for j in range(n):
        if swap:
            transform_simd(
                state,
                n - 1 - j,
                P(2 * pi / 2 ** (n - j) * v),
                GateKind.P,
                FloatType(2 * pi / 2 ** (n - j) * v),
                ctx,
            )
        else:       
            transform_simd(
                state,
                j,
                P(2 * pi / 2 ** (j + 1) * v),
                GateKind.P,
                FloatType(2 * pi / 2 ** (j + 1) * v),
                ctx,
            )

    return state^

fn iqft_simd(
    mut state: QuantumState,
    targets: List[Int],
    swap: Bool = False,
    ctx: ExecContext = ExecContext(),
):
    for j in reversed(range(len(targets))):
        # transform_h_simd(state, targets[j])
        transform_simd(state, targets[j], H, GateKind.H, ctx=ctx)
        for k in reversed(range(j)):
             # c_transform_p_simd(state, targets[j], targets[k], -pi / 2 ** (j - k))
            c_transform_simd(
                state,
                targets[j],
                targets[k],
                P(-pi / 2 ** (j - k)),
                GateKind.P,
                FloatType(-pi / 2 ** (j - k)),
                ctx,
            )
    # if swap:
    #     bit_reverse_state(state, True)

fn encode_value_simd(
    n: Int,
    v: FloatType,
    swap: Bool = False,
    ctx: ExecContext = ExecContext(),
) -> QuantumState:
    var state = prep_simd(n, v, swap, ctx)
    targets = [n - 1 - j for j in range(n)]
    if swap:
        targets = [j for j in range(n)]
    
    iqft_simd(state, targets, swap=swap, ctx=ctx)
    return state^


fn prep_simd_specialized(
    n: Int,
    v: FloatType,
    swap: Bool = False,
) -> QuantumState:
    var state = QuantumState(n)

    for j in range(n):
        transform_h_simd(state, j)

    for j in range(n):
        if swap:
            transform_p_simd(state, n - 1 - j, 2 * pi / 2 ** (n-j) * v)
        else:
            transform_p_simd(state, j, 2 * pi / 2 ** (j + 1) * v)

    return state^


fn iqft_simd_specialized(
    mut state: QuantumState,
    targets: List[Int],
    swap: Bool = False,
):
    for j in reversed(range(len(targets))):
        transform_h_simd(state, targets[j])
        for k in reversed(range(j)):
            c_transform_p_simd(state, targets[j], targets[k], -pi / 2 ** (j - k))
    _ = swap


fn encode_value_simd_specialized(
    n: Int,
    v: FloatType,
    swap: Bool = False,
) -> QuantumState:
    var state = prep_simd_specialized(n, v, swap)
    targets = [n - 1 - j for j in range(n)]
    if swap:
        targets = [j for j in range(n)]

    iqft_simd_specialized(state, targets, swap=swap)
    return state^

fn test_main() raises:
    from butterfly.utils.visualization import print_state

    var n = 3
    var v = 4.7

    swap = False
    var state = prep_simd(n, v, swap)
    print_state(state, short=False)

    # swap = True
    # state = prep_simd(n, v, swap)
    # print_state(state, short=False)

    swap = False
    state = encode_value_simd(n, v, swap)
    print_state(state, short=False)

    # swap = True
    # state = encode_value_simd(n, v, swap)
    # print_state(state, short=False)
