from butterfly.algos.shor_polynomial import build_shor_polynomial_circuit
from butterfly.algos.shor import (
    estimate_order_from_state,
    factors_from_order,
)
from butterfly.core.state import QuantumState
from butterfly.core.executors import execute
from butterfly.utils.context import ExecContext
from butterfly.utils.visualization import (
    FrameSource,
    animate_frame_source_pair,
)


# WARNING: This uses polynomial phase-encoding for f(x)=2^x mod N.
# It is a gate-only visualization approach, but it is not a reversible
# modular exponentiation circuit like Shor requires.


fn main() raises:
    test_animate_shor_poly_modexp()


fn test_animate_shor_poly_modexp() raises:
    var modulus = 15
    # var value_bits = 0
    # var tmp = modulus - 1
    # while tmp > 0:
    #     value_bits += 1
    #     tmp >>= 1
    # if value_bits == 0:
    #     value_bits = 1
    # var exp_bits = value_bits * 2
    var value_bits = 4  # Enough to hold 0..21
    var exp_bits = 4  # Small

    var qc = build_shor_polynomial_circuit(exp_bits, value_bits, modulus)

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
        step_on_input=True,
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
