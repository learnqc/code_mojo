from collections import List


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


fn tools_summary() -> String:
    var specs = tool_specs()
    var text = "Available tools:\n"
    for i in range(TOOL_COUNT):
        var spec = specs[i]
        text += "- " + spec.name + ": " + spec.description
        if i + 1 < TOOL_COUNT:
            text += "\n"
    return text


fn tools_json_schema() -> String:
    var specs = tool_specs()
    var out = "["
    for i in range(TOOL_COUNT):
        var spec = specs[i]
        if i > 0:
            out += ","
        out += "{"
        out += "\"name\":\"" + spec.name + "\","
        out += "\"description\":\"" + spec.description + "\","
        out += "\"parameters\":{"
        out += "\"type\":\"object\","
        out += "\"properties\":{"
        for j in range(spec.param_count):
            var p = spec.params[j]
            if j > 0:
                out += ","
            out += "\"" + p.name + "\":{"
            out += "\"type\":\"" + p.dtype + "\","
            out += "\"description\":\"" + p.description + "\""
            if p.has_minimum:
                out += ",\"minimum\":" + String(p.minimum)
            if p.has_items:
                out += ",\"items\":{\"type\":\"" + p.items_type + "\"}"
            out += "}"
        out += "},"
        out += "\"required\":["
        var first_req = True
        for j in range(spec.param_count):
            var p = spec.params[j]
            if p.required:
                if not first_req:
                    out += ","
                out += "\"" + p.name + "\""
                first_req = False
        out += "],"
        out += "\"additionalProperties\":false"
        out += "}"
        out += "}"
    out += "]"
    return out
