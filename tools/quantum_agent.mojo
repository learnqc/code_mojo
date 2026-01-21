from collections import List
from python import Python, PythonObject

from butterfly.core.executors import execute
from butterfly.core.quantum_circuit import QuantumCircuit
from butterfly.core.state import QuantumState
from butterfly.core.types import FloatType, pi
from butterfly.utils.context import ExecContext, ExecutionStrategy
from butterfly.utils.circuit_print import print_circuit_ascii
from butterfly.utils.visualization import (
    print_state,
    print_state_grid_colored_cells,
)

from tools.tool_spec import tools_summary
from tools.openai_provider import (
    OpenAIProvider,
    parse_openai_args,
    tools_openai_object,
)

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


fn require_int(token: String) raises -> Int:
    try:
        return Int(token)
    except:
        raise Error("Invalid integer: " + token)


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


fn py_list_len(obj: PythonObject) raises -> Int:
    var builtins = Python.import_module("builtins")
    return Int(builtins.len(obj))


fn sanitize_token(token: String) -> String:
    var t = String(token.strip())
    while t.startswith("("):
        t = String(t[1:])
    while t.endswith(")") or t.endswith(","):
        t = String(t[: len(t) - 1])
    return String(t)


fn looks_like_angle(token: String) -> Bool:
    if token.find("pi") >= 0:
        return True
    if token.find(".") >= 0:
        return True
    return False


fn looks_like_index(token: String) -> Bool:
    var t = String(token.strip())
    if t.startswith("-"):
        return False
    for i in range(len(t)):
        var c = t[i]
        if c < "0" or c > "9":
            return False
    return String(t) != ""


fn parse_bool_from_string(value: String, default: Bool) -> Bool:
    var val = value.lower()
    if val == "true" or val == "1" or val == "yes" or val == "on":
        return True
    if val == "false" or val == "0" or val == "no" or val == "off":
        return False
    return default


fn parse_args_list(raw: String) -> List[String]:
    var parts = raw.split(" ")
    var out = List[String]()
    for i in range(len(parts)):
        var token = String(parts[i]).strip()
        if String(token) != "":
            out.append(String(token))
    return out^


fn get_arg(
    args: List[Tuple[String, String]],
    key: String,
) -> Optional[String]:
    for i in range(len(args)):
        var (k, v) = args[i]
        if k == key:
            return Optional[String](v)
    return None


fn get_arg_list(
    args: List[Tuple[String, String]],
    key: String,
) -> List[String]:
    var out = List[String]()
    var raw = get_arg(args, key)
    if raw:
        return parse_args_list(raw.value())
    return out^


fn execute_tool(
    mut session: Session,
    name: String,
    args: List[Tuple[String, String]],
) raises -> Tuple[String, Bool]:
    if name == "create_circuit":
        var qubits_val = get_arg(args, "qubits")
        if not qubits_val:
            qubits_val = get_arg(args, "N")
        var qubits = 0
        if qubits_val:
            qubits = require_int_flexible(qubits_val.value())
        if qubits <= 0:
            raise Error("Invalid qubit count.")
        session.circuit = QuantumCircuit(qubits)
        session.has_circuit = True
        return ("Created circuit with " + String(qubits) + " qubits.", False)
    if name == "add_gate":
        var gate_val = get_arg(args, "gate")
        if not gate_val:
            raise Error("Missing gate name.")
        var gate = gate_val.value().lower()
        var arg_list = get_arg_list(args, "args")
        for i in range(len(arg_list)):
            arg_list[i] = sanitize_token(arg_list[i])
        if (gate == "p" or gate == "rx" or gate == "ry" or gate == "rz") and len(arg_list) == 2:
            if looks_like_angle(arg_list[0]) and looks_like_index(arg_list[1]):
                var tmp = arg_list[0]
                arg_list[0] = arg_list[1]
                arg_list[1] = tmp
        apply_gate(session, gate, arg_list)
        return ("Added gate: " + gate, False)
    if name == "show_circuit":
        if not ensure_circuit(session):
            return ("No circuit.", False)
        print_circuit_ascii(session.circuit)
        return ("Displayed circuit.", True)
    if name == "show_state":
        if not ensure_circuit(session):
            return ("No circuit.", False)
        var state = compute_state(session)
        print_state(state, short=False, use_color=True)
        return ("Displayed state.", True)
    if name == "show_grid":
        if not ensure_circuit(session):
            return ("No circuit.", False)
        var col_bits = 2
        var col_bits_val = get_arg(args, "col_bits")
        if col_bits_val:
            col_bits = require_int_flexible(col_bits_val.value())
        var use_log = True
        var log_val = get_arg(args, "log")
        if log_val:
            use_log = parse_bool_from_string(log_val.value(), True)
        var show_bin = True
        var bin_val = get_arg(args, "bin")
        if bin_val:
            show_bin = parse_bool_from_string(bin_val.value(), True)
        var state = compute_state(session)
        print_state_grid_colored_cells(
            state,
            col_bits,
            use_log=use_log,
            show_bin_labels=show_bin,
        )
        return ("Displayed grid.", True)
    if name == "list_tools":
        return (tools_summary(), True)
    raise Error("Unknown tool: " + name)


fn build_messages() raises -> PythonObject:
    var builtins = Python.import_module("builtins")
    var messages = builtins.list()
    var sys_msg = builtins.dict()
    sys_msg.__setitem__("role", value="system")
    sys_msg.__setitem__(
        "content",
        value=(
            "You control a quantum circuit simulator. Always use the provided "
            "tools; never answer without tool results and never invent tool "
            "names. Use create_circuit for new circuits, add_gate for gates, "
            "show_circuit when asked to show the circuit, show_state when asked "
            "to show state, show_grid when asked for a grid, and list_tools when "
            "asked about tools. If the user requests both circuit changes and "
            "a display, call the tools in order (changes first, then show). "
            "Do not call show tools unless the user requests it."
        ),
    )
    messages.append(sys_msg)
    return messages


fn append_user_message(messages: PythonObject, text: String) raises:
    var builtins = Python.import_module("builtins")
    var msg = builtins.dict()
    msg.__setitem__("role", value="user")
    msg.__setitem__("content", value=text)
    messages.append(msg)


fn append_tool_message(
    messages: PythonObject,
    tool_call_id: String,
    content: String,
) raises:
    var builtins = Python.import_module("builtins")
    var msg = builtins.dict()
    msg.__setitem__("role", value="tool")
    msg.__setitem__("tool_call_id", value=tool_call_id)
    msg.__setitem__("content", value=content)
    messages.append(msg)


fn ensure_ascii(label: String, value: String) raises -> Bool:
    var builtins = Python.import_module("builtins")
    try:
        var py_str = builtins.str(value)
        py_str.encode("ascii")
        return True
    except:
        print("Error: " + label + " must be ASCII.")
        return False


fn main() raises:
    from sys import argv

    var os = Python.import_module("os")
    var json = Python.import_module("json")
    var builtins = Python.import_module("builtins")

    var base_url = String(
        os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    var model = String(os.getenv("OPENAI_MODEL", "gpt-4.1"))
    var api_key = String(os.getenv("OPENAI_API_KEY", ""))
    var require_tools = True

    for i in range(len(argv())):
        var arg = String(argv()[i])
        if arg == "--base-url" and i + 1 < len(argv()):
            base_url = String(argv()[i + 1])
        if arg == "--model" and i + 1 < len(argv()):
            model = String(argv()[i + 1])
        if arg == "--no-require-tools":
            require_tools = False

    if api_key == "":
        print("Error: OPENAI_API_KEY is not set.")
        return
    if not ensure_ascii("OPENAI_API_KEY", api_key):
        return
    if not ensure_ascii("OPENAI_MODEL", model):
        return
    if not ensure_ascii("OPENAI_BASE_URL", base_url):
        return

    var tools = tools_openai_object()
    var messages = build_messages()
    var provider = OpenAIProvider(base_url, model, api_key)
    var session = Session()

    var sys = Python.import_module("sys")
    var stdin = sys.stdin
    print("Quantum agent ready. Type 'quit' or 'exit' to leave.")

    while True:
        print("> ", end="")
        var line_obj = stdin.readline()
        if line_obj is None:
            break
        var line = String(String(line_obj).strip())
        if line == "":
            continue
        if line == "quit" or line == "exit":
            break

        append_user_message(messages, line)
        var message = provider.chat(messages, tools, require_tools)
        var tool_calls = message.get("tool_calls")
        if tool_calls is None or py_list_len(tool_calls) == 0:
            print("No tool calls returned.")
            continue

        messages.append(message)
        var count = py_list_len(tool_calls)
        for i in range(count):
            var call = tool_calls.__getitem__(i)
            var func = call.__getitem__("function")
            var name = String(func.__getitem__("name"))
            var args_str = String(func.__getitem__("arguments"))
            var args_obj = json.loads(args_str)
            var args = parse_openai_args(args_obj)
            var call_id = String(call.__getitem__("id"))
            var (content, should_print) = execute_tool(session, name, args)
            if should_print:
                print(content)
            append_tool_message(messages, call_id, content)
