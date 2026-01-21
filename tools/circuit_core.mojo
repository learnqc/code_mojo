from collections import List

from butterfly.core.executors import execute
from butterfly.core.quantum_circuit import QuantumCircuit
from butterfly.core.state import QuantumState
from butterfly.core.types import FloatType, pi
from butterfly.utils.context import ExecContext, ExecutionStrategy


struct Session:
    var has_circuit: Bool
    var circuit: QuantumCircuit
    var strategy: Int
    var threads: Int

    fn __init__(out self):
        self.has_circuit = False
        self.circuit = QuantumCircuit(0)
        self.strategy = ExecutionStrategy.SCALAR
        self.threads = 0


fn parse_angle(expr: String) raises -> FloatType:
    var trimmed = String(expr.strip())
    if trimmed.find("pi") >= 0:
        var s = trimmed.replace(" ", "")
        var sign = FloatType(1.0)
        if s.startswith("-"):
            sign = FloatType(-1.0)
            s = String(s[1:])
        if s == "pi":
            return sign * pi
        if s.endswith("*pi"):
            var coeff_str = String(s[: len(s) - 3])
            var coeff = FloatType(Float64(coeff_str))
            return sign * coeff * pi
        if s.startswith("pi/"):
            var denom_str = String(s[3:])
            var denom = FloatType(Float64(denom_str))
            return sign * pi / denom
        if s.find("/pi") >= 0:
            var parts = s.split("/pi")
            if len(parts) == 2:
                var coeff = FloatType(Float64(String(parts[0])))
                return sign * coeff / pi
    var no_space = trimmed.replace(" ", "")
    if no_space.find("/") >= 0:
        var parts = no_space.split("/")
        if len(parts) == 2:
            var num = Float64(String(parts[0]))
            var denom = Float64(String(parts[1]))
            return FloatType(num / denom)
    return FloatType(Float64(trimmed))


fn require_int_flexible(token: String) raises -> Int:
    try:
        return Int(token)
    except:
        try:
            var f = Float64(token)
            return Int(f)
        except:
            raise Error("Invalid integer: " + token)


fn ensure_circuit(session: Session) -> Bool:
    if not session.has_circuit:
        print("Error: create a circuit first (create <n>).")
        return False
    return True


fn compute_state(session: Session) raises -> QuantumState:
    var state = QuantumState(session.circuit.num_qubits)
    var ctx = ExecContext()
    ctx.execution_strategy = session.strategy
    ctx.threads = session.threads
    execute(state, session.circuit, ctx)
    return state^


fn compute_state_for_circuit(
    session: Session,
    circuit: QuantumCircuit,
) raises -> QuantumState:
    var state = QuantumState(session.circuit.num_qubits)
    var ctx = ExecContext()
    ctx.execution_strategy = session.strategy
    ctx.threads = session.threads
    execute(state, circuit, ctx)
    return state^


fn apply_gate(mut session: Session, name: String, args: List[String]) raises:
    if not ensure_circuit(session):
        return
    var circuit = session.circuit.copy()
    if name == "h":
        circuit.h(require_int_flexible(args[0]))
    elif name == "x":
        circuit.x(require_int_flexible(args[0]))
    elif name == "y":
        circuit.y(require_int_flexible(args[0]))
    elif name == "z":
        circuit.z(require_int_flexible(args[0]))
    elif name == "p":
        circuit.p(require_int_flexible(args[0]), parse_angle(args[1]))
    elif name == "rx":
        circuit.rx(require_int_flexible(args[0]), parse_angle(args[1]))
    elif name == "ry":
        circuit.ry(require_int_flexible(args[0]), parse_angle(args[1]))
    elif name == "rz":
        circuit.rz(require_int_flexible(args[0]), parse_angle(args[1]))
    elif name == "cx":
        circuit.cx(require_int_flexible(args[0]), require_int_flexible(args[1]))
    elif name == "cy":
        circuit.cy(require_int_flexible(args[0]), require_int_flexible(args[1]))
    elif name == "cz":
        circuit.cz(require_int_flexible(args[0]), require_int_flexible(args[1]))
    elif name == "cp":
        circuit.cp(
            require_int_flexible(args[0]),
            require_int_flexible(args[1]),
            parse_angle(args[2]),
        )
    elif name == "crx":
        circuit.crx(
            require_int_flexible(args[0]),
            require_int_flexible(args[1]),
            parse_angle(args[2]),
        )
    elif name == "cry":
        circuit.cry(
            require_int_flexible(args[0]),
            require_int_flexible(args[1]),
            parse_angle(args[2]),
        )
    elif name == "crz":
        circuit.crz(
            require_int_flexible(args[0]),
            require_int_flexible(args[1]),
            parse_angle(args[2]),
        )
    elif name == "ccx":
        circuit.ccx(
            require_int_flexible(args[0]),
            require_int_flexible(args[1]),
            require_int_flexible(args[2]),
        )
    elif name == "swap":
        circuit.swap(require_int_flexible(args[0]), require_int_flexible(args[1]))
    elif name == "mcp":
        var n = len(args)
        if n < 3:
            raise Error("mcp requires at least 3 args.")
        var controls = List[Int](capacity=n - 2)
        for i in range(n - 2):
            controls.append(require_int_flexible(args[i]))
        var target = require_int_flexible(args[n - 2])
        var theta = parse_angle(args[n - 1])
        circuit.mcp(controls, target, theta)
    else:
        raise Error("Unknown gate: " + name)
    session.circuit = circuit^
