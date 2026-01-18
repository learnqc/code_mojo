from collections import List

from butterfly.algos.function_encoding import build_polynomial_circuit
from butterfly.core.state import QuantumState
from butterfly.core.types import FloatType
from butterfly.utils.visualization import (
    FrameSource,
    animate_frame_source_pair,
)


fn modexp_int(base: Int, exp: Int, modulus: Int) -> Int:
    if modulus <= 1:
        return 0
    var result = 1 % modulus
    var b = base % modulus
    var e = exp
    while e > 0:
        if (e & 1) == 1:
            result = (result * b) % modulus
        b = (b * b) % modulus
        e >>= 1
    return result


fn build_exponential_terms(n: Int, scale: FloatType) -> List[Tuple[FloatType, List[Int]]]:
    # Exact multilinear expansion for 2^x on n bits: product over (1 + (2^{2^i}-1) b_i).
    # WARNING: term count is 2^n, so keep n small.
    var terms = List[Tuple[FloatType, List[Int]]]()
    var total = 1 << n
    for mask in range(total):
        var coeff = scale
        var vars = List[Int]()
        for i in range(n):
            if ((mask >> i) & 1) == 1:
                var factor = FloatType((2 ** (2 ** i)) - 1)
                coeff *= factor
                vars.append(i)
        terms.append((coeff, vars^))
    return terms^


fn build_modexp_terms(
    n: Int,
    modulus: Int,
    scale: FloatType,
) -> List[Tuple[FloatType, List[Int]]]:
    # Exact multilinear expansion from the truth table f(x)=2^x mod modulus.
    # Uses Mobius transform; term count is 2^n.
    var total = 1 << n
    var coeffs = List[FloatType](length=total, fill=FloatType(0))
    for x in range(total):
        coeffs[x] = FloatType(modexp_int(2, x, modulus)) * scale
    for i in range(n):
        for mask in range(total):
            if ((mask >> i) & 1) == 1:
                coeffs[mask] -= coeffs[mask ^ (1 << i)]
    var terms = List[Tuple[FloatType, List[Int]]]()
    for mask in range(total):
        var vars = List[Int]()
        for i in range(n):
            if ((mask >> i) & 1) == 1:
                vars.append(i)
        terms.append((coeffs[mask], vars^))
    return terms^


fn main() raises:
    test_animate_exponential_encoding()


fn test_animate_exponential_encoding() raises:
    var key_size = 4
    var value_size = 4
    var max_y = (1 << value_size) - 1
    var use_mod = True
    var modulus = 21

    var terms: List[Tuple[FloatType, List[Int]]]
    if use_mod:
        var max_val = FloatType(modulus - 1)
        var scale = 1 #FloatType(max_y) / max_val
        terms = build_modexp_terms(key_size, modulus, scale)
    else:
        var max_x = (1 << key_size) - 1
        var max_val = FloatType(2 ** max_x)
        var scale = 1 #FloatType(max_y) / max_val
        terms = build_exponential_terms(key_size, scale)

    var qc = build_polynomial_circuit(key_size, value_size, terms)
    var left_state = QuantumState(key_size + value_size)
    var right_state = QuantumState(key_size + value_size)

    var left_source = FrameSource.grid(
        qc.copy(),
        left_state,
        col_bits=key_size,
        use_log=True,
        origin_bottom=True,
        show_bin_labels=True,
        use_bg=True,
        show_chars=False,
        show_step_label=True,
    )
    var right_source = FrameSource.table(
        qc.copy(),
        right_state,
        show_step_label=True,
        row_separators=True,
        max_rows=16,
    )

    animate_frame_source_pair(
        left_source,
        right_source,
        gap=6,
        delay_s=0.2,
        step_on_input=False,
    )
