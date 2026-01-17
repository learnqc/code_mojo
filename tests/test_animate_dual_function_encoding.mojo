from collections import List

from butterfly.algos.function_encoding import build_polynomial_circuit
from butterfly.core.state import QuantumState
from butterfly.core.types import FloatType
from butterfly.utils.visualization import (
    FrameSource,
    animate_frame_source_pair,
)


fn main() raises:
    test_animate_dual_function_encoding()

fn test_animate_dual_function_encoding() raises:
    # Parabola: y = scale * x^2, x encoded by key bits.
    var key_size = 3  
    var value_size = 4  
    var max_x = (1 << key_size) - 1
    var max_y = (1 << value_size) - 1
    var scale = FloatType(0.25) #FloatType(max_y) / FloatType(max_x * max_x)

    var terms = List[Tuple[FloatType, List[Int]]]()
    for i in range(key_size):
        var coeff = scale * FloatType(2 ** (2 * i))
        var vars = List[Int]()
        vars.append(i)
        terms.append((coeff, vars^))
        for j in range(i + 1, key_size):
            var coeff_ij = scale * FloatType(2 * (2 ** (i + j)))
            var vars_ij = List[Int]()
            vars_ij.append(i)
            vars_ij.append(j)
            terms.append((coeff_ij, vars_ij^))

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
        delay_s=0.1,
        step_on_input=False,
    )
