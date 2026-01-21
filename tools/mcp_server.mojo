from collections import List
from python import Python, PythonObject

from tools.circuit_core import Session, compute_state, ensure_circuit
from tools.quantum_agent import execute_tool
from tools.tool_spec import (
    MAX_TOOL_PARAMS,
    ParamSpec,
    TOOL_COUNT,
    tool_specs,
    tools_summary,
)
from butterfly.utils.circuit_print import circuit_to_ascii
from butterfly.utils.visualization import (
    render_grid_frame_lines,
    render_table_frame_lines,
)


fn build_mcp_tools() raises -> PythonObject:
    var builtins = Python.import_module("builtins")
    var tools = builtins.list()
    var specs = tool_specs()
    for i in range(TOOL_COUNT):
        var spec = specs[i]
        var tool = builtins.dict()
        tool.__setitem__("name", value=spec.name)
        tool.__setitem__("description", value=spec.description)
        tool.__setitem__("inputSchema", value=build_input_schema(spec.params, spec.param_count))
        tools.append(tool)
    return tools


fn build_input_schema(
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


fn parse_args_map(args: PythonObject) raises -> List[Tuple[String, String]]:
    var builtins = Python.import_module("builtins")
    var keys = builtins.list(args.keys())
    var count = Int(builtins.len(keys))
    var out = List[Tuple[String, String]](capacity=count)
    for i in range(count):
        var key = String(keys.__getitem__(i))
        var value = args.__getitem__(key)
        if builtins.isinstance(value, builtins.list):
            var items = builtins.list(value)
            var parts = List[String]()
            for j in range(Int(builtins.len(items))):
                parts.append(String(items.__getitem__(j)))
            var joined = ""
            for j in range(len(parts)):
                if j > 0:
                    joined += " "
                joined += parts[j]
            out.append((key, joined))
        else:
            out.append((key, String(value)))
    return out^


fn join_lines(lines: List[String]) -> String:
    var out = ""
    for i in range(len(lines)):
        if i > 0:
            out += "\n"
        out += lines[i]
    return out


fn parse_bool(value: String, default: Bool) -> Bool:
    var v = value.lower()
    if v == "true" or v == "1" or v == "yes" or v == "on":
        return True
    if v == "false" or v == "0" or v == "no" or v == "off":
        return False
    return default


fn render_circuit(session: Session) -> String:
    if not ensure_circuit(session):
        return "No circuit."
    return circuit_to_ascii(session.circuit) + "\nDisplayed circuit."


fn render_state_table(session: Session) raises -> String:
    if not ensure_circuit(session):
        return "No circuit."
    var state = compute_state(session)
    var lines = render_table_frame_lines(
        state,
        0,
        0,
        "",
        False,
        True,
        False,
        left_pad=0,
    )
    for i in range(len(lines)):
        lines[i] = lines[i].replace("￤", "|")
    var out = join_lines(lines)
    return out + "\nDisplayed state."


fn render_state_grid(
    session: Session,
    col_bits: Int,
    use_log: Bool,
    show_bin: Bool,
) raises -> String:
    if not ensure_circuit(session):
        return "No circuit."
    var state = compute_state(session)
    var lines = render_grid_frame_lines(
        state,
        0,
        0,
        "",
        col_bits,
        use_log,
        True,
        True,
        show_bin,
        False,
        False,
        left_pad=0,
    )
    var out = join_lines(lines)
    return out + "\nDisplayed grid."


fn json_rpc_response(id: PythonObject, result: PythonObject) raises -> PythonObject:
    var builtins = Python.import_module("builtins")
    var resp = builtins.dict()
    resp.__setitem__("jsonrpc", value="2.0")
    resp.__setitem__("id", value=id)
    resp.__setitem__("result", value=result)
    return resp


fn json_rpc_error(id: PythonObject, code: Int, message: String) raises -> PythonObject:
    var builtins = Python.import_module("builtins")
    var err = builtins.dict()
    err.__setitem__("code", value=code)
    err.__setitem__("message", value=message)
    var resp = builtins.dict()
    resp.__setitem__("jsonrpc", value="2.0")
    resp.__setitem__("id", value=id)
    resp.__setitem__("error", value=err)
    return resp


fn handle_json_rpc(
    request: PythonObject,
    mut session: Session,
    tools: PythonObject,
) raises -> PythonObject:
    var builtins = Python.import_module("builtins")
    var method = String(request.get("method"))
    var id_val = request.get("id")
    if id_val is None:
        id_val = builtins.None

    if method == "initialize":
        var info = builtins.dict()
        info.__setitem__("name", value="quantum-mcp")
        info.__setitem__("version", value="0.1")
        var caps = builtins.dict()
        caps.__setitem__("tools", value=builtins.dict())
        var result = builtins.dict()
        result.__setitem__("serverInfo", value=info)
        result.__setitem__("capabilities", value=caps)
        return json_rpc_response(id_val, result)

    if method == "tools/list":
        var result = builtins.dict()
        result.__setitem__("tools", value=tools)
        return json_rpc_response(id_val, result)

    if method == "tools/call":
        var params = request.get("params")
        if params is None:
            return json_rpc_error(id_val, -32602, "Missing params.")
        var name = String(params.get("name"))
        var arguments = params.get("arguments")
        if arguments is None:
            arguments = builtins.dict()
        var args = parse_args_map(arguments)
        var content = ""
        var is_error = False
        try:
            if name == "show_circuit":
                content = render_circuit(session)
            elif name == "show_state":
                content = render_state_table(session)
            elif name == "show_grid":
                var col_bits = 2
                var use_log = True
                var show_bin = True
                for i in range(len(args)):
                    var (k, v) = args[i]
                    if k == "col_bits":
                        col_bits = Int(String(v))
                    if k == "log":
                        use_log = parse_bool(String(v), True)
                    if k == "bin":
                        show_bin = parse_bool(String(v), True)
                content = render_state_grid(session, col_bits, use_log, show_bin)
            elif name == "list_tools":
                content = tools_summary()
            else:
                var (text, _) = execute_tool(session, name, args)
                content = text
        except:
            is_error = True
            content = "Tool failed."
        var text_part = builtins.dict()
        text_part.__setitem__("type", value="text")
        text_part.__setitem__("text", value=content)
        var content_list = builtins.list()
        content_list.append(text_part)
        var result = builtins.dict()
        result.__setitem__("content", value=content_list)
        result.__setitem__("isError", value=is_error)
        return json_rpc_response(id_val, result)

    return json_rpc_error(id_val, -32601, "Method not found.")


fn read_http_request(conn: PythonObject) raises -> Optional[String]:
    var builtins = Python.import_module("builtins")
    var data = ""
    var header_end = -1
    var content_length = 0

    while True:
        var chunk = conn.recv(4096)
        if chunk is None:
            break
        var text = String(chunk.decode("utf-8"))
        if text == "":
            break
        data += text
        header_end = data.find("\r\n\r\n")
        if header_end >= 0:
            var header_text = String(data[:header_end])
            var lines = header_text.split("\r\n")
            for i in range(len(lines)):
                var line = String(lines[i])
                var lower = line.lower()
                if lower.startswith("content-length:"):
                    var parts = line.split(":")
                    if len(parts) > 1:
                        content_length = Int(String(parts[1]).strip())
            var body = String(data[header_end + 4:])
            if len(body) >= content_length:
                break
        if len(data) > 1024 * 1024:
            break

    if data == "":
        return None
    return Optional[String](data)


fn write_http_response(conn: PythonObject, status: String, body: String) raises:
    var header = "HTTP/1.1 " + status + "\r\n"
    header += "Content-Type: application/json\r\n"
    header += "Content-Length: " + String(len(body)) + "\r\n"
    header += "Connection: close\r\n\r\n"
    var payload = header + body
    var builtins = Python.import_module("builtins")
    var py_bytes = builtins.bytes(builtins.str(payload), "utf-8")
    conn.sendall(py_bytes)


fn main() raises:
    from sys import argv

    var os = Python.import_module("os")
    var json = Python.import_module("json")
    var socket = Python.import_module("socket")

    var host = String(os.getenv("MCP_HOST", "127.0.0.1"))
    var port = Int(os.getenv("MCP_PORT", "8765"))
    for i in range(len(argv())):
        var arg = String(argv()[i])
        if arg == "--host" and i + 1 < len(argv()):
            host = String(argv()[i + 1])
        if arg == "--port" and i + 1 < len(argv()):
            port = Int(String(argv()[i + 1]))

    var server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    var builtins = Python.import_module("builtins")
    var addr = builtins.list()
    addr.append(host)
    addr.append(port)
    server.bind(builtins.tuple(addr))
    server.listen(5)

    var tools = build_mcp_tools()
    var session = Session()
    print("MCP server listening on http://" + host + ":" + String(port))

    while True:
        var conn = server.accept().__getitem__(0)
        var raw_opt = read_http_request(conn)
        if not raw_opt:
            conn.close()
            continue
        var raw = raw_opt.value()
        var header_end = raw.find("\r\n\r\n")
        if header_end < 0:
            write_http_response(conn, "400 Bad Request", "{\"error\":\"bad request\"}")
            conn.close()
            continue
        var headers = String(raw[:header_end])
        var first_line = String(headers.split("\r\n")[0])
        if first_line.startswith("GET") and first_line.find("/health") >= 0:
            var body = "{\"status\":\"ok\"}"
            write_http_response(conn, "200 OK", body)
            conn.close()
            continue

        var body = String(raw[header_end + 4:])
        if body == "":
            write_http_response(conn, "400 Bad Request", "{\"error\":\"missing body\"}")
            conn.close()
            continue
        var request = json.loads(body)
        var response = handle_json_rpc(request, session, tools)
        var response_text = String(json.dumps(response))
        write_http_response(conn, "200 OK", response_text)
        conn.close()
