from butterfly.algos.value_encoding_circuit import encode_value_circuit
from butterfly.core.state import QuantumState
from butterfly.utils.visualization import animate_execution, animate_execution_table


fn main() raises:
    var n = 5
    var circuit = encode_value_circuit(n, 4.7)
    var state = QuantumState(n)
    animate_execution_table(
        circuit,
        state,
        show_step_label=True,
        redraw_in_place=True,
        delay_s=0.25,
        step_on_input=True
    )
    # animate_execution(
    #     circuit,
    #     state,
    #     col_bits=2,
    #     use_log=True,
    #     show_bin_labels=True,
    #     use_bg=True,
    #     show_chars=False,
    #     redraw_in_place=True,
    #     delay_s=0.25,
    #     step_on_input=False
    # )
