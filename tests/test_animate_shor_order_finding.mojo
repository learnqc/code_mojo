from butterfly.algos.shor import (
    order_finding_circuit,
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


fn main() raises:
    test_animate_shor_order_finding()


fn test_animate_shor_order_finding() raises:
    # Shor order-finding demo for N=15 with a=2.
    var modulus = 15
    var a = 2
    var value_bits = 4  # Enough to hold 0..15
    var exp_bits = 4  # Small demo size

    var qc = order_finding_circuit(exp_bits, value_bits, a, modulus)
    var state = QuantumState(exp_bits + value_bits)

    var left_source = FrameSource.grid(
        qc.copy(),
        state.copy(),
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
        state,
        exp_bits,
        value_bits,
        show_step_label=True,
        row_separators=True,
        max_rows=16,
    )

    animate_frame_source_pair(
        left_source,
        right_source,
        gap=6,
        delay_s=0.25,
        step_on_input=False,
    )

    var final_state = QuantumState(exp_bits + value_bits)
    execute(final_state, qc, ExecContext())
    var r = estimate_order_from_state(
        final_state,
        exp_bits,
        value_bits,
        a,
        modulus,
    )
    if r:
        print("Estimated order r = " + String(r.value()))
        var factors = factors_from_order(a, modulus, r.value())
        if factors:
            print(
                "Factors of " 
                + String(modulus)
                + " = "
                + String(factors.value()[0])
                + " * "
                + String(factors.value()[1])
            )
        else:
            print("Factors not found from r.")
    else:
        print("Estimated order not found.")
