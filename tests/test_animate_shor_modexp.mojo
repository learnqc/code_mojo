from butterfly.algos.shor import modexp_circuit
from butterfly.core.state import QuantumState
from butterfly.utils.visualization import (
    FrameSource,
    animate_frame_source_pair,
)


fn main() raises:
    test_animate_shor_modexp()


fn test_animate_shor_modexp() raises:
    # Visualize only the modular exponentiation encoding (no IQFT).
    var modulus = 15
    var a = 2
    var value_bits = 4
    var exp_bits = 4

    var qc = modexp_circuit(exp_bits, value_bits, a, modulus)
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
    var right_source = FrameSource.table(
        qc.copy(),
        state,
        show_step_label=True,
        row_separators=True,
        max_rows=16,
    )

    animate_frame_source_pair(
        left_source,
        right_source,
        gap=6,
        delay_s=0.5,
        step_on_input=True,
    )
