from butterfly.algos.value_encoding_circuit import encode_value_circuit
from butterfly.core.state import QuantumState
from butterfly.core.types import FloatType
from butterfly.utils.visualization import (
    FrameSource,
    animate_frame_source_pair,
)


fn main() raises:
    var n = 4
    var v_left = FloatType(4.7)
    var v_right = FloatType(4.7)

    var circuit_left = encode_value_circuit(n, v_left)
    var circuit_right = encode_value_circuit(n, v_right)

    var left_state = QuantumState(n)
    var right_state = QuantumState(n)

    var left_source = FrameSource.grid(
        circuit_left.copy(),
        left_state,
        col_bits=2,
        use_log=True,
        show_bin_labels=True,
        use_bg=True,
        show_chars=False,
        show_step_label=True,
    )
    var right_source = FrameSource.table(
        circuit_right.copy(),
        right_state,
        show_step_label=True,
        row_separators=False,
    )

    # Left width is auto-derived from the left visualization.
    animate_frame_source_pair(
        left_source,
        right_source,
        gap=6,
        delay_s=0.75,
        step_on_input=True
    )
