from butterfly.core.state import QuantumState, generate_state, apply_bit_reverse
from butterfly.core.types import FloatType


fn assert_state_close(
    a: QuantumState,
    b: QuantumState,
    tol: FloatType,
) raises:
    if a.size() != b.size():
        raise Error("State size mismatch")
    for i in range(a.size()):
        var dr = a.re[i] - b.re[i]
        var di = a.im[i] - b.im[i]
        if dr < 0:
            dr = -dr
        if di < 0:
            di = -di
        if dr > tol or di > tol:
            raise Error("State mismatch at " + String(i))


fn test_full_bit_reverse_matches_partial() raises:
    print("Testing full bit-reverse matches partial targets...")
    alias n = 4
    var full_targets = List[Int](0, 1, 2, 3)
    var seeds = List[Int](5, 11, 17)
    for seed in seeds:
        var state = generate_state(n, seed)
        var a = state.copy()
        var b = state.copy()
        apply_bit_reverse(a, List[Int]())
        apply_bit_reverse(b, full_targets)
        assert_state_close(a, b, FloatType(1e-6))


fn test_bit_reverse_is_involution() raises:
    print("Testing bit-reverse involution property...")
    alias n = 4
    var full_targets = List[Int](0, 1, 2, 3)
    var subset_targets = List[Int](1, 3)
    var seeds = List[Int](7, 13, 19)
    for seed in seeds:
        var state = generate_state(n, seed)
        var a = state.copy()
        var b = state.copy()
        apply_bit_reverse(a, List[Int]())
        apply_bit_reverse(a, List[Int]())
        assert_state_close(a, state, FloatType(1e-6))

        apply_bit_reverse(b, full_targets)
        apply_bit_reverse(b, full_targets)
        assert_state_close(b, state, FloatType(1e-6))

        var c = state.copy()
        apply_bit_reverse(c, subset_targets)
        apply_bit_reverse(c, subset_targets)
        assert_state_close(c, state, FloatType(1e-6))


fn main() raises:
    test_full_bit_reverse_matches_partial()
    test_bit_reverse_is_involution()
    print("Bit-reverse tests passed!")
