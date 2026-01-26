from collections import List
from python import Python, PythonObject

from tools.tool_spec import (
    MAX_TOOL_PARAMS,
    ParamSpec,
    TOOL_COUNT,
    tool_specs,
)


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


fn parse_openai_args(args: PythonObject) raises -> List[Tuple[String, String]]:
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
        var raw: String
        try:
            var resp = urllib.urlopen(req, timeout=60)
            raw = String(resp.read().decode("utf-8"))
        except:
            raise Error("HTTP error from OpenAI.")
        var response = json.loads(raw)
        var choices = response.__getitem__("choices")
        var first = choices.__getitem__(0)
        return first.__getitem__("message")


fn build_openai_messages() raises -> PythonObject:
    var builtins = Python.import_module("builtins")
    var messages = builtins.list()
    return messages


fn append_openai_system_message(messages: PythonObject, text: String) raises:
    var builtins = Python.import_module("builtins")
    var msg = builtins.dict()
    msg.__setitem__("role", value="system")
    msg.__setitem__("content", value=text)
    messages.append(msg)


fn append_openai_user_message(messages: PythonObject, text: String) raises:
    var builtins = Python.import_module("builtins")
    var msg = builtins.dict()
    msg.__setitem__("role", value="user")
    msg.__setitem__("content", value=text)
    messages.append(msg)


fn append_openai_tool_message(
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


fn extract_openai_tool_calls(message: PythonObject) raises -> PythonObject:
    return message.get("tool_calls")


fn get_openai_tool_call_name(call: PythonObject) raises -> String:
    var func = call.__getitem__("function")
    return String(func.__getitem__("name"))


fn get_openai_tool_call_args_json(call: PythonObject) raises -> String:
    var func = call.__getitem__("function")
    return String(func.__getitem__("arguments"))


fn get_openai_tool_call_id(call: PythonObject) raises -> String:
    var call_id = call.get("id")
    if call_id is None:
        return "tool-call"
    return String(call_id)
