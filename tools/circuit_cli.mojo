from collections import List

from butterfly.core.executors import execute
from butterfly.core.quantum_circuit import (
    QuantumCircuit,
    bit_reverse,
    measure,
    permute_qubits,
)
from butterfly.core.state import QuantumState
from butterfly.core.types import FloatType, pi
from butterfly.utils.context import ExecContext, ExecutionStrategy
from butterfly.utils.circuit_print import print_circuit_ascii
from butterfly.utils.visualization import (
    print_state,
    print_state_grid_colored_cells,
)

struct Session:
    var has_circuit: Bool
    var circuit: QuantumCircuit
    var num_qubits: Int
    var strategy: Int
    var threads: Int

    fn __init__(out self):
        self.has_circuit = False
        self.circuit = QuantumCircuit(0)
        self.num_qubits = 0
        self.strategy = ExecutionStrategy.SCALAR
        self.threads = 0


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


fn split_tokens(line: String) -> List[String]:
    var raw = line.split(" ")
    var tokens = List[String]()
    for item in raw:
        var token = String(String(item).strip())
        if token != "":
            tokens.append(token)
    return tokens^


fn parse_int(token: String) -> Optional[Int]:
    try:
        return Optional[Int](Int(token))
    except:
        return None


fn require_int(token: String) raises -> Int:
    var parsed = parse_int(token)
    if parsed:
        return parsed.value()
    raise Error("Invalid integer: " + token)


fn ensure_circuit(session: Session) -> Bool:
    if not session.has_circuit:
        print("Error: create a circuit first (create <n>).")
        return False
    return True


fn compute_state(session: Session) raises -> QuantumState:
    var state = QuantumState(session.num_qubits)
    var ctx = ExecContext()
    ctx.execution_strategy = session.strategy
    ctx.threads = session.threads
    execute(state, session.circuit, ctx)
    return state^


fn compute_state_for_circuit(
    session: Session,
    circuit: QuantumCircuit,
) raises -> QuantumState:
    var state = QuantumState(session.num_qubits)
    var ctx = ExecContext()
    ctx.execution_strategy = session.strategy
    ctx.threads = session.threads
    execute(state, circuit, ctx)
    return state^


fn render_grid_in_place(
    state: QuantumState,
    col_bits: Int,
    use_log: Bool,
    show_bin: Bool,
) raises:
    print("\033[2J\033[H", end="")
    print_state_grid_colored_cells(
        state,
        col_bits,
        use_log=use_log,
        show_bin_labels=show_bin,
    )


fn print_help():
    print("Commands:")
    print("  create <n>")
    print("  reset | clear")
    print("  strategy <scalar|scalar_parallel|simd|simd_parallel|grid|grid_parallel|grid_fused|grid_parallel_fused>")
    print("  threads <n>")
    print("  h|x|y|z <target>")
    print("  p|rx|ry|rz <target> <angle>  (angle supports pi, pi/2, 0.5*pi, etc)")
    print("  cx|cy|cz <control> <target>")
    print("  cp|crx|cry|crz <control> <target> <angle>")
    print("  ccx <control1> <control2> <target>")
    print("  mcp <c1> <c2> ... <target> <angle>")
    print("  swap <a> <b>")
    print("  measure <t1> <t2> ...")
    print("  bitrev [t1 t2 ...]")
    print("  qrev [t1 t2 ...]")
    print("  permute <i0> <i1> ...")
    print("  show circuit")
    print("  show state")
    print("  show grid <col_bits> [--log] [--bin]")
    print("  show steps <col_bits> [--log] [--bin]")
    print("  help")
    print("  quit | exit")


fn apply_gate(mut session: Session, name: String, args: List[String]) raises:
    if not ensure_circuit(session):
        return
    var circuit = session.circuit.copy()
    if name == "h":
        circuit.h(require_int(args[0]))
    elif name == "x":
        circuit.x(require_int(args[0]))
    elif name == "y":
        circuit.y(require_int(args[0]))
    elif name == "z":
        circuit.z(require_int(args[0]))
    elif name == "p":
        circuit.p(require_int(args[0]), parse_angle(args[1]))
    elif name == "rx":
        circuit.rx(require_int(args[0]), parse_angle(args[1]))
    elif name == "ry":
        circuit.ry(require_int(args[0]), parse_angle(args[1]))
    elif name == "rz":
        circuit.rz(require_int(args[0]), parse_angle(args[1]))
    elif name == "cx":
        circuit.cx(require_int(args[0]), require_int(args[1]))
    elif name == "cy":
        circuit.cy(require_int(args[0]), require_int(args[1]))
    elif name == "cz":
        circuit.cz(require_int(args[0]), require_int(args[1]))
    elif name == "cp":
        circuit.cp(
            require_int(args[0]),
            require_int(args[1]),
            parse_angle(args[2]),
        )
    elif name == "crx":
        circuit.crx(
            require_int(args[0]),
            require_int(args[1]),
            parse_angle(args[2]),
        )
    elif name == "cry":
        circuit.cry(
            require_int(args[0]),
            require_int(args[1]),
            parse_angle(args[2]),
        )
    elif name == "crz":
        circuit.crz(
            require_int(args[0]),
            require_int(args[1]),
            parse_angle(args[2]),
        )
    elif name == "ccx":
        circuit.ccx(
            require_int(args[0]),
            require_int(args[1]),
            require_int(args[2]),
        )
    elif name == "swap":
        circuit.swap(require_int(args[0]), require_int(args[1]))
    elif name == "mcp":
        var n = len(args)
        if n < 3:
            raise Error("mcp requires at least 3 args.")
        var controls = List[Int](capacity=n - 2)
        for i in range(n - 2):
            controls.append(require_int(args[i]))
        var target = require_int(args[n - 2])
        var theta = parse_angle(args[n - 1])
        circuit.mcp(controls, target, theta)
    else:
        raise Error("Unknown gate: " + name)
    session.circuit = circuit^


fn apply_command(mut session: Session, line: String) raises -> Bool:
    var tokens = split_tokens(line)
    if len(tokens) == 0:
        return True
    var cmd = String(tokens[0]).lower()
    var args = List[String]()
    if len(tokens) > 1:
        for i in range(1, len(tokens)):
            args.append(String(tokens[i]))

    if cmd == "quit" or cmd == "exit":
        return False
    if cmd == "help":
        print_help()
        return True
    if cmd == "create":
        if len(args) != 1:
            print("Usage: create <n>")
            return True
        var n = 0
        try:
            n = require_int(args[0])
        except:
            print("Invalid qubit count: " + args[0])
            return True
        session.circuit = QuantumCircuit(n)
        session.num_qubits = n
        session.has_circuit = True
        print("Created circuit with " + String(n) + " qubits.")
        return True
    if cmd == "reset" or cmd == "clear":
        if not ensure_circuit(session):
            return True
        session.circuit = QuantumCircuit(session.num_qubits)
        print("Circuit cleared.")
        return True
    if cmd == "strategy":
        if len(args) != 1:
            print("Usage: strategy <name>")
            return True
        var name = args[0].lower()
        if name == "scalar":
            session.strategy = ExecutionStrategy.SCALAR
        elif name == "scalar_parallel":
            session.strategy = ExecutionStrategy.SCALAR_PARALLEL
        elif name == "simd":
            session.strategy = ExecutionStrategy.SIMD
        elif name == "simd_parallel":
            session.strategy = ExecutionStrategy.SIMD_PARALLEL
        elif name == "grid":
            session.strategy = ExecutionStrategy.GRID
        elif name == "grid_parallel":
            session.strategy = ExecutionStrategy.GRID_PARALLEL
        elif name == "grid_fused":
            session.strategy = ExecutionStrategy.GRID_FUSED
        elif name == "grid_parallel_fused":
            session.strategy = ExecutionStrategy.GRID_PARALLEL_FUSED
        else:
            print("Unknown strategy: " + name)
            return True
        print("Strategy set: " + name)
        return True
    if cmd == "threads":
        if len(args) != 1:
            print("Usage: threads <n>")
            return True
        try:
            session.threads = require_int(args[0])
        except:
            print("Invalid thread count: " + args[0])
        return True
    if cmd == "show":
        if len(args) == 0:
            print("Usage: show <circuit|state|grid>")
            return True
        var sub = args[0].lower()
        if sub == "circuit":
            if not ensure_circuit(session):
                return True
            print_circuit_ascii(session.circuit)
            return True
        if sub == "state":
            if not ensure_circuit(session):
                return True
            try:
                var state = compute_state(session)
                print_state(state, short=False, use_color=True)
            except e:
                print("Error: " + String(e))
            return True
        if sub == "grid":
            if not ensure_circuit(session):
                return True
            if len(args) < 2:
                print("Usage: show grid <col_bits> [--log] [--bin]")
                return True
            var col_bits = 0
            try:
                col_bits = require_int(args[1])
            except:
                print("Invalid col_bits: " + args[1])
                return True
            var use_log = False
            var show_bin = False
            for i in range(2, len(args)):
                if args[i] == "--log":
                    use_log = True
                elif args[i] == "--bin":
                    show_bin = True
            try:
                var state = compute_state(session)
                print_state_grid_colored_cells(
                    state,
                    col_bits,
                    use_log=use_log,
                    show_bin_labels=show_bin,
                )
            except e:
                print("Error: " + String(e))
            return True
        if sub == "steps":
            if not ensure_circuit(session):
                return True
            if len(args) < 2:
                print("Usage: show steps <col_bits> [--log] [--bin]")
                return True
            var col_bits = 0
            try:
                col_bits = require_int(args[1])
            except:
                print("Invalid col_bits: " + args[1])
                return True
            var use_log = False
            var show_bin = False
            for i in range(2, len(args)):
                if args[i] == "--log":
                    use_log = True
                elif args[i] == "--bin":
                    show_bin = True
            var total = len(session.circuit.transformations)
            if total == 0:
                print("Circuit has no transformations.")
                return True
            try:
                for i in range(total):
                    var sub_circuit = QuantumCircuit(session.num_qubits)
                    for j in range(i + 1):
                        sub_circuit.transformations.append(
                            session.circuit.transformations[j].copy()
                        )
                    var state = compute_state_for_circuit(session, sub_circuit)
                    render_grid_in_place(state, col_bits, use_log, show_bin)
            except e:
                print("Error: " + String(e))
            return True
        print("Unknown show option: " + sub)
        return True
    if cmd == "measure":
        if not ensure_circuit(session):
            return True
        if len(args) == 0:
            print("Usage: measure <t1> <t2> ...")
            return True
        var targets = List[Int](capacity=len(args))
        try:
            for a in args:
                targets.append(require_int(a))
        except e:
            print("Error: " + String(e))
            return True
        measure(session.circuit, targets)
        return True
    if cmd == "bitrev":
        if not ensure_circuit(session):
            return True
        if len(args) == 0:
            bit_reverse(session.circuit)
        else:
            var targets = List[Int](capacity=len(args))
            try:
                for a in args:
                    targets.append(require_int(a))
            except e:
                print("Error: " + String(e))
                return True
            bit_reverse(session.circuit, targets)
        return True
    if cmd == "qrev":
        if not ensure_circuit(session):
            return True
        if len(args) == 0:
            session.circuit.qubit_reversal()
        else:
            var targets = List[Int](capacity=len(args))
            try:
                for a in args:
                    targets.append(require_int(a))
            except e:
                print("Error: " + String(e))
                return True
            session.circuit.qubit_reversal(targets)
        return True
    if cmd == "permute":
        if not ensure_circuit(session):
            return True
        if len(args) == 0:
            print("Usage: permute <i0> <i1> ...")
            return True
        var order = List[Int](capacity=len(args))
        try:
            for a in args:
                order.append(require_int(a))
        except e:
            print("Error: " + String(e))
            return True
        permute_qubits(session.circuit, order)
        return True

    try:
        var is_gate = (
            cmd == "h"
            or cmd == "x"
            or cmd == "y"
            or cmd == "z"
            or cmd == "p"
            or cmd == "rx"
            or cmd == "ry"
            or cmd == "rz"
            or cmd == "cx"
            or cmd == "cy"
            or cmd == "cz"
            or cmd == "cp"
            or cmd == "crx"
            or cmd == "cry"
            or cmd == "crz"
            or cmd == "ccx"
            or cmd == "swap"
            or cmd == "mcp"
        )
        if is_gate:
            apply_gate(session, cmd, args)
            return True
    except e:
        print("Error: " + String(e))
        return True

    print("Unknown command: " + cmd)
    return True


fn main() raises:
    from python import Python
    from sys import argv

    var prompt = False
    for i in range(len(argv())):
        if argv()[i] == "--prompt":
            prompt = True

    var session = Session()
    var sys = Python.import_module("sys")
    var stdin = sys.stdin

    print("Butterfly circuit CLI. Type 'help' for commands.")
    while True:
        if prompt:
            print("> ", end="")
        var line_obj = stdin.readline()
        if line_obj is None:
            break
        var line = String(String(line_obj).strip())
        if line == "":
            continue
        var keep = True
        try:
            keep = apply_command(session, line)
        except e:
            print("Error: " + String(e))
            keep = True
        if not keep:
            break

# create 3
# h 0
# x 1
# cp 0 2 pi/4
# show circuit
# show state
# show grid 2 --log --bin
