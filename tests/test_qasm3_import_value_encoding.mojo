from collections import List

from butterfly.algos.value_encoding_circuit import encode_value_circuit
from butterfly.core.circuit import GateTransformation
from butterfly.core.executors import execute_scalar
from butterfly.core.gates import GateKind
from butterfly.core.quantum_circuit import QuantumCircuit
from butterfly.core.state import QuantumState
from butterfly.core.types import FloatType
from butterfly.utils.benchmark_verify import verify_states_equal
from butterfly.utils.qasm3_import import parse_qasm3_string


fn qasm_from_circuit(circuit: QuantumCircuit) raises -> String:
    var lines = List[String]()
    lines.append("OPENQASM 3;")
    lines.append("include \"stdgates.inc\";")
    lines.append("qubit[" + String(circuit.num_qubits) + "] q;")

    for tr in circuit.transformations:
        if not tr.isa[GateTransformation]():
            raise Error("Unsupported transformation for QASM export")
        var gt = tr[GateTransformation].copy()
        if gt.kind == 0 and gt.gate_info.kind == GateKind.H:
            lines.append("h q[" + String(gt.target) + "];")
        elif gt.kind == 0 and gt.gate_info.kind == GateKind.P:
            var angle = Float64(gt.gate_info.arg.value())
            lines.append(
                "p(" + String(angle) + ") q[" + String(gt.target) + "];"
            )
        elif gt.kind == 1 and gt.gate_info.kind == GateKind.P:
            var angle = Float64(gt.gate_info.arg.value())
            lines.append(
                "cp("
                + String(angle)
                + ") q["
                + String(gt.controls[0])
                + "], q["
                + String(gt.target)
                + "];"
            )
        else:
            raise Error("Unsupported gate for QASM export")

    var out = ""
    for i in range(len(lines)):
        out += lines[i] + "\n"
    return out^

from butterfly.utils.visualization import print_state
fn main() raises:
    alias n = 3
    var v = FloatType(4.7)
    var circuit = encode_value_circuit(n, v)

    var qasm = qasm_from_circuit(circuit)
    print(qasm)
    var imported = parse_qasm3_string(qasm)

    var state_orig = QuantumState(n)
    execute_scalar(state_orig, circuit)
    var state_imported = QuantumState(n)
    execute_scalar(state_imported, imported)
    print_state(state_imported)

    _ = verify_states_equal(
        state_orig,
        state_imported,
        tolerance=1e-10,
        name1="orig",
        name2="qasm3",
    )
