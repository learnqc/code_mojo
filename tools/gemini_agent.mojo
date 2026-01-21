from python import Python, PythonObject

from tools.quantum_agent import execute_tool
from tools.circuit_core import Session
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
    tools_gemini_object,
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


fn main() raises:
    from sys import argv

    var os = Python.import_module("os")

    var base_url = String(
        os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
    )
    var model = String(os.getenv("GEMINI_MODEL", "gemini-3-pro-preview"))
    var api_key = String(os.getenv("GEMINI_API_KEY", ""))
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
        print("Error: GEMINI_API_KEY is not set.")
        return
    if not ensure_ascii("GEMINI_API_KEY", api_key):
        return
    if not ensure_ascii("GEMINI_MODEL", model):
        return
    if not ensure_ascii("GEMINI_BASE_URL", base_url):
        return

    var provider = GeminiProvider(base_url, model, api_key)
    var list_models = String(os.getenv("GEMINI_LIST_MODELS", "")) == "1"
    if list_models:
        print(provider.list_models())
        return

    var tools = tools_gemini_object()
    var messages = build_messages()
    var session = Session()

    var sys = Python.import_module("sys")
    var stdin = sys.stdin
    print("Gemini quantum agent ready. Type 'quit' or 'exit' to leave.")

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
            var (content, should_print) = execute_tool(session, name, args)
            if should_print:
                print(content)
            append_gemini_tool_message(messages, name, content)
