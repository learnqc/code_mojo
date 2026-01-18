from butterfly.algos.shor_unitary import order_finding_unitary_circuit
from butterfly.core.state import QuantumState
from butterfly.utils.visualization import (
    FrameSource,
    animate_frame_source_pair,
)


fn main() raises:
    test_animate_shor_order_finding_unitary()


fn test_animate_shor_order_finding_unitary() raises:
    var modulus = 15
    var a = 2
    var value_bits = 4
    var exp_bits = 4

    var qc = order_finding_unitary_circuit(exp_bits, value_bits, a, modulus)
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
