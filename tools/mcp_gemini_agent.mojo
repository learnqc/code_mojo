from python import Python, PythonObject

from tools.gemini_provider import (
    GeminiProvider,
    append_gemini_system_message,
    append_gemini_tool_message,
    append_gemini_user_message,
    build_gemini_messages,
    extract_gemini_tool_calls,
    get_gemini_tool_call_args,
    get_gemini_tool_call_id,
    get_gemini_tool_call_name,
    parse_gemini_args,
)


fn py_list_len(obj: PythonObject) raises -> Int:
    var builtins = Python.import_module("builtins")
    return Int(builtins.len(obj))


fn ensure_ascii(label: String, value: String) raises -> Bool:
    var builtins = Python.import_module("builtins")
    try:
        var py_str = builtins.str(value)
        py_str.encode("ascii")
        return True
    except:
        print("Error: " + label + " must be ASCII.")
        return False


fn mcp_request(base_url: String, payload: PythonObject) raises -> PythonObject:
    var json = Python.import_module("json")
    var urllib = Python.import_module("urllib.request")
    var builtins = Python.import_module("builtins")

    var data = json.dumps(payload).encode("utf-8")
    var headers = builtins.dict()
    headers.__setitem__("Content-Type", value="application/json")
    var req = urllib.Request(
        base_url,
        data=data,
        headers=headers,
        method="POST",
    )
    var resp = urllib.urlopen(req, timeout=60)
    var raw = String(resp.read().decode("utf-8"))
    return json.loads(raw)


fn mcp_list_tools(base_url: String) raises -> PythonObject:
    var builtins = Python.import_module("builtins")
    var payload = builtins.dict()
    payload.__setitem__("jsonrpc", value="2.0")
    payload.__setitem__("id", value=1)
    payload.__setitem__("method", value="tools/list")
    payload.__setitem__("params", value=builtins.dict())
    var resp = mcp_request(base_url, payload)
    return resp.__getitem__("result").__getitem__("tools")


fn mcp_call_tool(
    base_url: String,
    name: String,
    args: PythonObject,
    call_id: Int,
) raises -> String:
    var builtins = Python.import_module("builtins")
    var params = builtins.dict()
    params.__setitem__("name", value=name)
    params.__setitem__("arguments", value=args)

    var payload = builtins.dict()
    payload.__setitem__("jsonrpc", value="2.0")
    payload.__setitem__("id", value=call_id)
    payload.__setitem__("method", value="tools/call")
    payload.__setitem__("params", value=params)

    var resp = mcp_request(base_url, payload)
    var result = resp.__getitem__("result")
    var content = result.__getitem__("content")
    var text = ""
    if content is not None:
        var parts = builtins.list(content)
        for i in range(Int(builtins.len(parts))):
            var part = parts.__getitem__(i)
            if part.get("type") == "text":
                text += String(part.get("text"))
    return text


fn convert_tools_for_gemini(tools: PythonObject) raises -> PythonObject:
    var builtins = Python.import_module("builtins")
    var decls = builtins.list()
    var count = Int(builtins.len(tools))
    for i in range(count):
        var tool = tools.__getitem__(i)
        var decl = builtins.dict()
        decl.__setitem__("name", value=tool.__getitem__("name"))
        decl.__setitem__("description", value=tool.__getitem__("description"))
        var schema = tool.__getitem__("inputSchema")
        var schema_copy = builtins.dict(schema)
        schema_copy.pop("additionalProperties", None)
        decl.__setitem__("parameters", value=schema_copy)
        decls.append(decl)
    var entry = builtins.dict()
    entry.__setitem__("functionDeclarations", value=decls)
    var out = builtins.list()
    out.append(entry)
    return out


fn build_messages() raises -> PythonObject:
    var messages = build_gemini_messages()
    append_gemini_system_message(
        messages,
        (
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
    return messages


fn args_to_python(args: List[Tuple[String, String]]) raises -> PythonObject:
    var builtins = Python.import_module("builtins")
    var out = builtins.dict()
    for i in range(len(args)):
        var (k, v) = args[i]
        if k == "args":
            var parts = v.split(" ")
            var items = builtins.list()
            for j in range(len(parts)):
                var part = String(parts[j]).strip()
                if String(part) != "":
                    items.append(String(part))
            out.__setitem__("args", value=items)
        else:
            out.__setitem__(k, value=v)
    return out


fn main() raises:
    from sys import argv

    var os = Python.import_module("os")

    var base_url = String(
        os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
    )
    var model = String(os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"))
    var api_key = String(os.getenv("GEMINI_API_KEY", ""))
    var require_tools = True

    var mcp_url = String(os.getenv("MCP_URL", "http://127.0.0.1:8765"))
    if not mcp_url.endswith("/mcp"):
        mcp_url += "/mcp"

    for i in range(len(argv())):
        var arg = String(argv()[i])
        if arg == "--base-url" and i + 1 < len(argv()):
            base_url = String(argv()[i + 1])
        if arg == "--model" and i + 1 < len(argv()):
            model = String(argv()[i + 1])
        if arg == "--no-require-tools":
            require_tools = False
        if arg == "--mcp-url" and i + 1 < len(argv()):
            mcp_url = String(argv()[i + 1])

    if api_key == "":
        print("Error: GEMINI_API_KEY is not set.")
        return
    if not ensure_ascii("GEMINI_API_KEY", api_key):
        return
    if not ensure_ascii("GEMINI_MODEL", model):
        return
    if not ensure_ascii("GEMINI_BASE_URL", base_url):
        return
    if not ensure_ascii("MCP_URL", mcp_url):
        return

    var mcp_tools = mcp_list_tools(mcp_url)
    var tools = convert_tools_for_gemini(mcp_tools)
    var messages = build_messages()
    var provider = GeminiProvider(base_url, model, api_key)

    var sys = Python.import_module("sys")
    var stdin = sys.stdin
    print("MCP Gemini agent ready. Type 'quit' or 'exit' to leave.")

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

        append_gemini_user_message(messages, line)
        var message = provider.chat(messages, tools, require_tools)
        var tool_calls = extract_gemini_tool_calls(message)
        if tool_calls is None or py_list_len(tool_calls) == 0:
            print("No tool calls returned.")
            continue

        messages.append(message)
        var count = py_list_len(tool_calls)
        for i in range(count):
            var call = tool_calls.__getitem__(i)
            var name = get_gemini_tool_call_name(call)
            var args_obj = get_gemini_tool_call_args(call)
            var args = parse_gemini_args(args_obj)
            var call_id = get_gemini_tool_call_id(i)
            var mcp_args = args_to_python(args)
            var content = mcp_call_tool(mcp_url, name, mcp_args, i + 1)
            if content != "":
                print(content)
            append_gemini_tool_message(messages, name, content)
