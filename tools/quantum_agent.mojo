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


alias MAX_TOOL_PARAMS: Int = 3
alias TOOL_COUNT: Int = 6


struct ParamSpec(Copyable, ImplicitlyCopyable, Movable):
    var name: String
    var dtype: String
    var description: String
    var required: Bool
    var minimum: Int
    var has_minimum: Bool
    var items_type: String
    var has_items: Bool

    fn __init__(out self):
        self.name = ""
        self.dtype = ""
        self.description = ""
        self.required = False
        self.minimum = 0
        self.has_minimum = False
        self.items_type = ""
        self.has_items = False

    fn __init__(
        out self,
        name: String,
        dtype: String,
        description: String,
        required: Bool,
        minimum: Int = 0,
        has_minimum: Bool = False,
        items_type: String = "",
        has_items: Bool = False,
    ):
        self.name = name
        self.dtype = dtype
        self.description = description
        self.required = required
        self.minimum = minimum
        self.has_minimum = has_minimum
        self.items_type = items_type
        self.has_items = has_items


struct ToolSpec(Copyable, ImplicitlyCopyable, Movable):
    var name: String
    var description: String
    var params: InlineArray[ParamSpec, MAX_TOOL_PARAMS]
    var param_count: Int

    fn __init__(out self):
        self.name = ""
        self.description = ""
        self.params = build_param_array()
        self.param_count = 0

    fn __init__(
        out self,
        name: String,
        description: String,
        params: InlineArray[ParamSpec, MAX_TOOL_PARAMS],
        param_count: Int,
    ):
        self.name = name
        self.description = description
        self.params = params
        self.param_count = param_count


fn build_param_array() -> InlineArray[ParamSpec, MAX_TOOL_PARAMS]:
    var params = InlineArray[ParamSpec, MAX_TOOL_PARAMS](
        ParamSpec(),
        ParamSpec(),
        ParamSpec(),
    )
    return params^


fn tool_specs() -> InlineArray[ToolSpec, TOOL_COUNT]:
    var tools = InlineArray[ToolSpec, TOOL_COUNT](
        ToolSpec(),
        ToolSpec(),
        ToolSpec(),
        ToolSpec(),
        ToolSpec(),
        ToolSpec(),
    )

    var params = build_param_array()
    params[0] = ParamSpec(
        "qubits",
        "integer",
        "Number of qubits to create.",
        True,
        minimum=1,
        has_minimum=True,
    )
    tools[0] = ToolSpec(
            "create_circuit",
            "Create or reset a circuit with N qubits.",
            params,
            1,
        )

    params = build_param_array()
    params[0] = ParamSpec(
        "gate",
        "string",
        "Gate name, e.g. h, x, rz, cx, mcp.",
        True,
    )
    params[1] = ParamSpec(
        "args",
        "array",
        "Gate arguments as strings.",
        True,
        items_type="string",
        has_items=True,
    )
    tools[1] = ToolSpec(
            "add_gate",
            "Add a gate to the circuit.",
            params,
            2,
        )

    params = build_param_array()
    tools[2] = ToolSpec(
            "show_circuit",
            "Display the circuit as ASCII.",
            params,
            0,
        )

    params = build_param_array()
    tools[3] = ToolSpec(
            "show_state",
            "Display the current circuit state as a table.",
            params,
            0,
        )

    params = build_param_array()
    params[0] = ParamSpec(
        "col_bits",
        "integer",
        "Number of column bits for the grid.",
        True,
        minimum=1,
        has_minimum=True,
    )
    params[1] = ParamSpec(
        "log",
        "boolean",
        "Use log scale for amplitudes.",
        True,
    )
    params[2] = ParamSpec(
        "bin",
        "boolean",
        "Show binary axis labels.",
        True,
    )
    tools[4] = ToolSpec(
            "show_grid",
            "Display the state grid with layout options.",
            params,
            3,
        )

    params = build_param_array()
    tools[5] = ToolSpec(
            "list_tools",
            "List available tools and their descriptions.",
            params,
            0,
        )

    return tools^


fn build_parameters_schema(
    params: InlineArray[ParamSpec, MAX_TOOL_PARAMS],
    param_count: Int,
) raises -> PythonObject:
    var builtins = Python.import_module("builtins")
    var props = builtins.dict()
    var required = builtins.list()

    for i in range(param_count):
        var p = params[i]
        var prop = builtins.dict()
        prop.__setitem__("type", value=p.dtype)
        prop.__setitem__("description", value=p.description)
        if p.has_minimum:
            prop.__setitem__("minimum", value=p.minimum)
        if p.has_items:
            var items = builtins.dict()
            items.__setitem__("type", value=p.items_type)
            prop.__setitem__("items", value=items)
        props.__setitem__(p.name, value=prop)
        if p.required:
            required.append(p.name)

    var schema = builtins.dict()
    schema.__setitem__("type", value="object")
    schema.__setitem__("properties", value=props)
    schema.__setitem__("required", value=required)
    schema.__setitem__("additionalProperties", value=False)
    return schema


fn tools_openai_object() raises -> PythonObject:
    var builtins = Python.import_module("builtins")
    var tools = builtins.list()
    var specs = tool_specs()
    for i in range(TOOL_COUNT):
        var spec = specs[i]
        var func = builtins.dict()
        func.__setitem__("name", value=spec.name)
        func.__setitem__("description", value=spec.description)
        func.__setitem__(
            "parameters",
            value=build_parameters_schema(spec.params, spec.param_count),
        )
        var tool = builtins.dict()
        tool.__setitem__("type", value="function")
        tool.__setitem__("function", value=func)
        tools.append(tool)
    return tools


fn tools_json_schema() raises -> String:
    var builtins = Python.import_module("builtins")
    var json = Python.import_module("json")
    var tools = builtins.list()
    var specs = tool_specs()
    for i in range(TOOL_COUNT):
        var spec = specs[i]
        var obj = builtins.dict()
        obj.__setitem__("name", value=spec.name)
        obj.__setitem__("description", value=spec.description)
        obj.__setitem__(
            "parameters",
            value=build_parameters_schema(spec.params, spec.param_count),
        )
        tools.append(obj)
    return String(json.dumps(tools))


fn tools_summary() -> String:
    var specs = tool_specs()
    var text = "Available tools:\n"
    for i in range(TOOL_COUNT):
        var spec = specs[i]
        text += "- " + spec.name + ": " + spec.description
        if i + 1 < TOOL_COUNT:
            text += "\n"
    return text


struct OpenAIProvider:
    var base_url: String
    var model: String
    var api_key: String

    fn __init__(out self, base_url: String, model: String, api_key: String):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key

    fn chat(
        self,
        messages: PythonObject,
        tools: PythonObject,
        require_tools: Bool,
    ) raises -> PythonObject:
        var json = Python.import_module("json")
        var urllib = Python.import_module("urllib.request")
        var builtins = Python.import_module("builtins")

        var payload = builtins.dict()
        payload.__setitem__("model", value=self.model)
        payload.__setitem__("messages", value=messages)
        payload.__setitem__("tools", value=tools)
        if require_tools:
            payload.__setitem__("tool_choice", value="required")

        var headers = builtins.dict()
        headers.__setitem__("Content-Type", value="application/json")
        headers.__setitem__(
            "Authorization", value="Bearer " + self.api_key
        )

        var data = json.dumps(payload).encode("utf-8")
        var url = self.base_url.rstrip("/") + "/chat/completions"
        var req = urllib.Request(
            url,
            data=data,
            headers=headers,
            method="POST",
        )
        var raw = ""
        try:
            var resp = urllib.urlopen(req, timeout=60)
            raw = String(resp.read().decode("utf-8"))
        except:
            raise Error("HTTP error from OpenAI.")
        var response = json.loads(raw)
        var choices = response.__getitem__("choices")
        var first = choices.__getitem__(0)
        return first.__getitem__("message")


fn py_list_len(obj: PythonObject) raises -> Int:
    var builtins = Python.import_module("builtins")
    return Int(builtins.len(obj))


fn py_list_to_strings(obj: PythonObject) raises -> List[String]:
    var n = py_list_len(obj)
    var out = List[String](capacity=n)
    for i in range(n):
        out.append(String(obj.__getitem__(i)))
    return out^


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


fn py_get_bool(obj: PythonObject, default: Bool) raises -> Bool:
    if obj is None:
        return default
    var val = String(obj).lower()
    if val == "true" or val == "1":
        return True
    if val == "false" or val == "0":
        return False
    return default


fn py_get_int(obj: PythonObject, default: Int) raises -> Int:
    if obj is None:
        return default
    try:
        return Int(String(obj))
    except:
        return default


fn execute_tool(
    mut session: Session,
    name: String,
    args: PythonObject,
) raises -> Tuple[String, Bool]:
    if name == "create_circuit":
        var qubits_obj = args.get("qubits")
        if qubits_obj is None:
            qubits_obj = args.get("N")
        var qubits = py_get_int(qubits_obj, 0)
        if qubits <= 0:
            raise Error("Invalid qubit count.")
        session.circuit = QuantumCircuit(qubits)
        session.has_circuit = True
        return ("Created circuit with " + String(qubits) + " qubits.", False)
    if name == "add_gate":
        var gate = String(args.__getitem__("gate")).lower()
        var arg_list = py_list_to_strings(args.__getitem__("args"))
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
        var col_bits = py_get_int(args.get("col_bits"), 2)
        var use_log = py_get_bool(args.get("log"), True)
        var show_bin = py_get_bool(args.get("bin"), True)
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
            var args = json.loads(args_str)
            var call_id = String(call.__getitem__("id"))
            var (content, should_print) = execute_tool(session, name, args)
            if should_print:
                print(content)
            append_tool_message(messages, call_id, content)
