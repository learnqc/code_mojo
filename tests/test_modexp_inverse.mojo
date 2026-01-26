from testing import assert_almost_equal

from butterfly.algos.function_encoding import build_polynomial_circuit
from butterfly.algos.shor import apply_modexp, apply_modexp_inverse, modexp_circuit
from butterfly.algos.shor_polynomial import build_modexp_terms
from butterfly.core.executors import execute
from butterfly.core.state import QuantumState
from butterfly.core.types import FloatType
from butterfly.utils.context import ExecContext


fn build_state(n_qubits: Int) -> QuantumState:
    var state = QuantumState(n_qubits)
    var size = state.size()
    for i in range(size):
        state.re[i] = FloatType((i % 7) - 3) / 10.0
        state.im[i] = FloatType((i % 5) - 2) / 10.0
    state.invalidate_buffers()
    return state^


fn expected_re(index: Int) -> FloatType:
    return FloatType((index % 7) - 3) / 10.0


fn expected_im(index: Int) -> FloatType:
    return FloatType((index % 5) - 2) / 10.0


fn assert_state_eq(state: QuantumState) raises:
    for i in range(state.size()):
        assert_almost_equal(state.re[i], expected_re(i))
        assert_almost_equal(state.im[i], expected_im(i))

fn assert_states_close(left: QuantumState, right: QuantumState) raises:
    if left.size() != right.size():
        raise Error("State size mismatch in comparison")
    for i in range(left.size()):
        assert_almost_equal(left.re[i], right.re[i])
        assert_almost_equal(left.im[i], right.im[i])


fn test_modexp_inverse_roundtrip(n_exp: Int, n_value: Int, a: Int, modulus: Int) raises:
    var total = n_exp + n_value
    var state = build_state(total)
    var targets = List[Int](capacity=4)
    targets.append(n_exp)
    targets.append(n_value)
    targets.append(a)
    targets.append(modulus)

    apply_modexp(state, targets)
    apply_modexp_inverse(state, targets)
    assert_state_eq(state)

    var state2 = build_state(total)
    apply_modexp_inverse(state2, targets)
    apply_modexp(state2, targets)
    assert_state_eq(state2)

fn test_modexp_polynomial_matches_classical(
    n_exp: Int,
    n_value: Int,
    a: Int,
    modulus: Int,
) raises:
    var classical = modexp_circuit(n_exp, n_value, a, modulus)
    var terms = build_modexp_terms(n_exp, modulus, a)
    var poly = build_polynomial_circuit(n_exp, n_value, terms)

    var classical_state = QuantumState(n_exp + n_value)
    var poly_state = QuantumState(n_exp + n_value)
    execute(classical_state, classical, ExecContext())
    execute(poly_state, poly, ExecContext())
    assert_states_close(classical_state, poly_state)


fn main() raises:
    test_modexp_inverse_roundtrip(3, 3, 2, 5)
    test_modexp_inverse_roundtrip(4, 3, 2, 7)
    test_modexp_polynomial_matches_classical(3, 3, 2, 5)
    test_modexp_polynomial_matches_classical(4, 3, 2, 7)
