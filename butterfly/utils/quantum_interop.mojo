from butterfly.core.state import QuantumState
from python import PythonObject
from butterfly.utils.python_interop import python_to_float64

# --- Quantum Domain Interop Assembler ---
# This layer mediates between raw Python data and the QuantumState type.


fn state_from_python(data: PythonObject, n: Int) raises -> QuantumState:
    """Assembles a QuantumState from raw Python data (expected: [re_list, im_list]).
    """
    var re_list = data[0]
    var im_list = data[1]

    var size = 1 << n
    var state = QuantumState(n)

    for i in range(size):
        # Using Principle 89 logic via the interop helper
        state.re[i] = python_to_float64(re_list[i])
        state.im[i] = python_to_float64(im_list[i])

    return state^


fn get_qiskit_state(n: Int, value: Float64) raises -> QuantumState:
    """High-level domain helper to fetch a Qiskit state."""
    from butterfly.utils.python_interop import python_call

    var raw_data = python_call(
        "butterfly.utils.external_benchmarks", "get_qiskit_statevector_data", n, value
    )
    return state_from_python(raw_data, n)
