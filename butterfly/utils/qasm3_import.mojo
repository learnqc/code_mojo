from collections import List

from butterfly.core.quantum_circuit import QuantumCircuit
from butterfly.core.types import FloatType, pi


fn parse_angle(expr: String) raises -> FloatType:
    var trimmed = String(expr.strip())
    if trimmed.find("pi") >= 0:
        var s = trimmed.replace(" ", "")
        var sign = 1.0
        if s.startswith("-"):
            sign = -1.0
            s = String(s[1:])
        if s == "pi":
            return FloatType(sign * Float64(pi))
        if s.endswith("*pi"):
            var coeff_str = String(s[: len(s) - 3])
            var coeff = Float64(coeff_str)
            return FloatType(sign * coeff * Float64(pi))
        if s.startswith("pi/"):
            var denom_str = String(s[3:])
            var denom = Float64(denom_str)
            return FloatType(sign * Float64(pi) / denom)
        if s.find("/pi") >= 0:
            var parts = s.split("/pi")
            if len(parts) == 2:
                var coeff = Float64(String(parts[0]))
                return FloatType(sign * coeff / Float64(pi))
    return FloatType(Float64(trimmed))


fn cleanup_token(token: String) -> String:
    var t = String(token.strip())
    if t.endswith(";"):
        t = String(t[:-1].strip())
    if t.endswith(","):
        t = String(t[:-1].strip())
    return String(t)


fn parse_qubit_index(token: String) raises -> Int:
    var t = cleanup_token(token)
    var lb = t.find("[")
    var rb = t.find("]")
    if lb < 0 or rb < 0 or rb <= lb:
        raise Error("Invalid qubit token: " + t)
    var idx_str = String(t[lb + 1 : rb].strip())
    return Int(idx_str)


fn parse_qasm3_string(qasm: String) raises -> QuantumCircuit:
    var lines = qasm.split("\n")
    var n_qubits = -1
    var circuit = QuantumCircuit(0)
    var has_circuit = False

    for i in range(len(lines)):
        var s = String(lines[i]).strip()
        if s == "":
            continue
        if s.startswith("//"):
            continue
        if s.startswith("OPENQASM"):
            continue
        if s.startswith("include"):
            continue
        if s.startswith("qubit"):
            var lb = s.find("[")
            var rb = s.find("]")
            if lb < 0 or rb < 0:
                raise Error("Invalid qubit declaration: " + s)
            var n_str = String(s[lb + 1 : rb].strip())
            n_qubits = Int(n_str)
            circuit = QuantumCircuit(n_qubits)
            has_circuit = True
            continue

        if not has_circuit:
            raise Error("QASM missing qubit declaration before gates.")

        if s.endswith(";"):
            s = s[:-1].strip()

        if s.startswith("h "):
            var target = parse_qubit_index(String(s[2:]))
            circuit.h(target)
            continue

        if s.startswith("x "):
            var target = parse_qubit_index(String(s[2:]))
            circuit.x(target)
            continue

        if s.startswith("p("):
            var close = s.find(")")
            if close < 0:
                raise Error("Invalid p gate: " + s)
            var angle_str = String(s[2:close].strip())
            var target = parse_qubit_index(String(s[close + 1 :]))
            circuit.p(target, parse_angle(angle_str))
            continue

        if s.startswith("cp("):
            var close = s.find(")")
            if close < 0:
                raise Error("Invalid cp gate: " + s)
            var angle_str = String(s[3:close].strip())
            var rest = String(s[close + 1 :].strip())
            var parts = rest.split(",")
            if len(parts) != 2:
                raise Error("Invalid cp operands: " + rest)
            var control = parse_qubit_index(String(parts[0]))
            var target = parse_qubit_index(String(parts[1]))
            circuit.cp(control, target, parse_angle(angle_str))
            continue

        raise Error("Unsupported QASM line: " + s)

    if not has_circuit:
        raise Error("No qubit declaration found in QASM.")

    return circuit^
