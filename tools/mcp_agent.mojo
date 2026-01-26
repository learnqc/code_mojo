from collections import List
from collections import List
from python import Python, PythonObject

from tools.openai_provider import (
    OpenAIProvider,
    append_openai_system_message,
    append_openai_tool_message,
    append_openai_user_message,
    build_openai_messages,
    extract_openai_tool_calls,
    get_openai_tool_call_args_json,
    get_openai_tool_call_id,
    get_openai_tool_call_name,
    parse_openai_args,
)
from tools.cli_input import read_line_with_history


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
    var os = Python.import_module("os")
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
    args: List[Tuple[String, String]],
    call_id: Int,
) raises -> String:
    var builtins = Python.import_module("builtins")
    var params = builtins.dict()
    params.__setitem__("name", value=name)
    var arguments = builtins.dict()
    for i in range(len(args)):
        var (k, v) = args[i]
        arguments.__setitem__(k, value=v)
    params.__setitem__("arguments", value=arguments)

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


fn convert_tools_for_openai(tools: PythonObject) raises -> PythonObject:
    var builtins = Python.import_module("builtins")
    var out = builtins.list()
    var count = Int(builtins.len(tools))
    for i in range(count):
        var tool = tools.__getitem__(i)
        var func = builtins.dict()
        func.__setitem__("name", value=tool.__getitem__("name"))
        func.__setitem__("description", value=tool.__getitem__("description"))
        func.__setitem__("parameters", value=tool.__getitem__("inputSchema"))
        var entry = builtins.dict()
        entry.__setitem__("type", value="function")
        entry.__setitem__("function", value=func)
        out.append(entry)
    return out


fn build_messages() raises -> PythonObject:
    var messages = build_openai_messages()
    append_openai_system_message(
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


fn main() raises:
    from sys import argv

    var json = Python.import_module("json")

    var base_url = String(os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    var model = String(os.getenv("OPENAI_MODEL", "gpt-4.1"))
    var api_key = String(os.getenv("OPENAI_API_KEY", ""))
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
        print("Error: OPENAI_API_KEY is not set.")
        return
    if not ensure_ascii("OPENAI_API_KEY", api_key):
        return
    if not ensure_ascii("OPENAI_MODEL", model):
        return
    if not ensure_ascii("OPENAI_BASE_URL", base_url):
        return
    if not ensure_ascii("MCP_URL", mcp_url):
        return

    var mcp_tools = mcp_list_tools(mcp_url)
    var tools = convert_tools_for_openai(mcp_tools)
    var messages = build_messages()
    var provider = OpenAIProvider(base_url, model, api_key)

    var sys = Python.import_module("sys")
    print("MCP OpenAI agent ready. Type 'quit' or 'exit' to leave.")
    var history = List[String]()

    while True:
        var line_opt = read_line_with_history("> ", history)
        if not line_opt:
            break
        var line = String(String(line_opt.value()).strip())
        if line == "":
            continue
        if line == "quit" or line == "exit":
            break

        append_openai_user_message(messages, line)
        var message = provider.chat(messages, tools, require_tools)
        var tool_calls = extract_openai_tool_calls(message)
        if tool_calls is None or py_list_len(tool_calls) == 0:
            print("No tool calls returned.")
            continue

        messages.append(message)
        var count = py_list_len(tool_calls)
        for i in range(count):
            var call = tool_calls.__getitem__(i)
            var name = get_openai_tool_call_name(call)
            var args_obj = json.loads(get_openai_tool_call_args_json(call))
            var args = parse_openai_args(args_obj)
            var call_id = get_openai_tool_call_id(call)
            var content = mcp_call_tool(mcp_url, name, args, i + 1)
            if content != "":
                print(content)
            append_openai_tool_message(messages, call_id, content)
