from collections import List
from collections import List
from python import Python, PythonObject
from tools.cli_input import read_line_with_history


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
    return resp


fn mcp_call_tool(
    base_url: String,
    name: String,
    arguments: PythonObject,
    call_id: Int,
) raises -> PythonObject:
    var builtins = Python.import_module("builtins")
    var params = builtins.dict()
    params.__setitem__("name", value=name)
    params.__setitem__("arguments", value=arguments)

    var payload = builtins.dict()
    payload.__setitem__("jsonrpc", value="2.0")
    payload.__setitem__("id", value=call_id)
    payload.__setitem__("method", value="tools/call")
    payload.__setitem__("params", value=params)

    return mcp_request(base_url, payload)


fn print_tools(response: PythonObject) raises:
    var builtins = Python.import_module("builtins")
    if response.get("error") is not None:
        print("Error: " + String(response.__getitem__("error").__getitem__("message")))
        return
    var result = response.__getitem__("result")
    var tools = result.__getitem__("tools")
    var count = Int(builtins.len(tools))
    if count == 0:
        print("No tools returned.")
        return
    print("Available tools:")
    for i in range(count):
        var tool = tools.__getitem__(i)
        print("- " + String(tool.__getitem__("name")) + ": " + String(tool.__getitem__("description")))


fn print_tool_response(response: PythonObject) raises:
    var builtins = Python.import_module("builtins")
    if response.get("error") is not None:
        var err = response.__getitem__("error")
        print("Error: " + String(err.__getitem__("message")))
        return
    var result = response.__getitem__("result")
    var content = result.get("content")
    if content is None:
        print("No content returned.")
        return
    var parts = builtins.list(content)
    var output = ""
    for i in range(Int(builtins.len(parts))):
        var part = parts.__getitem__(i)
        if part.get("type") == "text":
            output += String(part.get("text"))
    if output != "":
        print(output)


fn parse_line(line: String) -> List[String]:
    var tokens = line.split(" ")
    var out = List[String]()
    for i in range(len(tokens)):
        var token = String(tokens[i]).strip()
        if String(token) != "":
            out.append(String(token))
    return out^


fn main() raises:
    from sys import argv

    var os = Python.import_module("os")
    var builtins = Python.import_module("builtins")

    var mcp_url = String(os.getenv("MCP_URL", "http://127.0.0.1:8765"))
    if not mcp_url.endswith("/mcp"):
        mcp_url += "/mcp"
    for i in range(len(argv())):
        var arg = String(argv()[i])
        if arg == "--mcp-url" and i + 1 < len(argv()):
            mcp_url = String(argv()[i + 1])

    if not ensure_ascii("MCP_URL", mcp_url):
        return

    var sys = Python.import_module("sys")
    print("MCP CLI ready. Type 'help' for commands.")
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
        if line == "help":
            print("Commands:")
            print("- list")
            print("- call <tool> [key=value ...] (repeat args= for list args)")
            print("- quit/exit")
            continue

        var parts = parse_line(line)
        if len(parts) == 0:
            continue
        var cmd = parts[0]
        if cmd == "list":
            var resp = mcp_list_tools(mcp_url)
            print_tools(resp)
            continue
        if cmd == "call":
            if len(parts) < 2:
                print("Usage: call <tool> [key=value ...]")
                continue
            var name = parts[1]
            var args = builtins.dict()
            var args_list = builtins.list()
            for i in range(2, len(parts)):
                var token = parts[i]
                var eq = token.find("=")
                if eq < 0:
                    continue
                var key = String(token[:eq])
                var value = String(token[eq + 1:])
                if key == "args":
                    args_list.append(value)
                else:
                    args.__setitem__(key, value=value)
            if Int(builtins.len(args_list)) > 0:
                args.__setitem__("args", value=args_list)
            var resp = mcp_call_tool(mcp_url, name, args, 1)
            print_tool_response(resp)
            continue
        var name = cmd
        var args = builtins.dict()
        var args_list = builtins.list()
        for i in range(1, len(parts)):
            var token = parts[i]
            var eq = token.find("=")
            if eq < 0:
                continue
            var key = String(token[:eq])
            var value = String(token[eq + 1:])
            if key == "args":
                args_list.append(value)
            else:
                args.__setitem__(key, value=value)
        if Int(builtins.len(args_list)) > 0:
            args.__setitem__("args", value=args_list)
        var resp = mcp_call_tool(mcp_url, name, args, 1)
        print_tool_response(resp)
        continue

        print("Unknown command. Type 'help' for commands.")
