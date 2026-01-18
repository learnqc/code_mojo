from collections import List

from butterfly.algos.function_encoding import build_polynomial_circuit
from butterfly.algos.shor import (
    estimate_order_from_state,
    factors_from_order,
)
from butterfly.algos.value_encoding_circuit import iqft_circuit
from butterfly.core.state import QuantumState
from butterfly.core.types import FloatType
from butterfly.core.quantum_circuit import QuantumCircuit, QuantumRegister
from butterfly.core.executors import execute
from butterfly.utils.context import ExecContext
from butterfly.utils.visualization import (
    FrameSource,
    animate_frame_source_pair,
)


# WARNING: This uses polynomial phase-encoding for f(x)=2^x mod N.
# It is a gate-only visualization approach, but it is not a reversible
# modular exponentiation circuit like Shor requires.


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


fn build_modexp_terms(
    n: Int,
    modulus: Int,
) -> List[Tuple[FloatType, List[Int]]]:
    # Exact multilinear expansion from the truth table f(x)=2^x mod modulus.
    var total = 1 << n
    var coeffs = List[FloatType](length=total, fill=FloatType(0))
    for x in range(total):
        coeffs[x] = FloatType(modexp_int(2, x, modulus))
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
    test_animate_shor_poly_modexp()


fn test_animate_shor_poly_modexp() raises:
    var modulus = 21
    # var value_bits = 0
    # var tmp = modulus - 1
    # while tmp > 0:
    #     value_bits += 1
    #     tmp >>= 1
    # if value_bits == 0:
    #     value_bits = 1
    # var exp_bits = value_bits * 2
    var value_bits = 1  # Enough to hold 0..21
    var exp_bits = 5  # Small

    var terms = build_modexp_terms(exp_bits, modulus)
    var qc = build_polynomial_circuit(exp_bits, value_bits, terms)

    # Apply IQFT on exponent register to mimic order-finding readout.
    var iqft = QuantumCircuit(exp_bits)
    _ = iqft_circuit(iqft, [exp_bits - 1 - j for j in range(exp_bits)], swap=False)
    _ = qc.append_circuit(iqft, QuantumRegister("exp", exp_bits))

    var left_state = QuantumState(exp_bits + value_bits)
    var right_state = QuantumState(exp_bits + value_bits)

    var left_source = FrameSource.grid(
        qc.copy(),
        left_state,
        col_bits=exp_bits,
        use_log=True,
        origin_bottom=True,
        show_bin_labels=True,
        use_bg=True,
        show_chars=False,
        show_step_label=True,
    )
    var right_source = FrameSource.exp_marginal_table(
        qc.copy(),
        right_state,
        exp_bits,
        value_bits,
        show_step_label=True,
        row_separators=True,
        max_rows=8,
    )

    animate_frame_source_pair(
        left_source,
        right_source,
        gap=6,
        delay_s=0.2,
        step_on_input=False,
    )

    var final_state = QuantumState(exp_bits + value_bits)
    execute(final_state, qc, ExecContext())
    var r = estimate_order_from_state(
        final_state,
        exp_bits,
        value_bits,
        2,
        modulus,
    )
    if r:
        print("Estimated order r = " + String(r.value()))
        var factors = factors_from_order(2, modulus, r.value())
        if factors:
            print(
                "Factors for " + String(modulus) + " = "
                + String(factors.value()[0])
                + " * "
                + String(factors.value()[1])
            )
        else:
            print("Factors not found from r.")
    else:
        print("Estimated order not found.")
