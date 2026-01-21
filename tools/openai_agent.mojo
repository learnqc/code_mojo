from python import Python, PythonObject

from tools.quantum_agent import execute_tool
from tools.circuit_core import Session
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
    tools_openai_object,
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

    var os = Python.import_module("os")
    var json = Python.import_module("json")

    var base_url = String(os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
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
    var debug = String(os.getenv("OPENAI_DEBUG", "")) == "1"

    var sys = Python.import_module("sys")
    var stdin = sys.stdin
    print("OpenAI quantum agent ready. Type 'quit' or 'exit' to leave.")

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

        append_openai_user_message(messages, line)
        var message = provider.chat(messages, tools, require_tools)
        var tool_calls = extract_openai_tool_calls(message)
        if tool_calls is None or py_list_len(tool_calls) == 0:
            if debug:
                print("Raw OpenAI message: " + String(json.dumps(message)))
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
            var (content, should_print) = execute_tool(session, name, args)
            if should_print:
                print(content)
            append_openai_tool_message(messages, call_id, content)
