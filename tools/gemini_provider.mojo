from collections import List
from python import Python, PythonObject
from sys import is_defined
from sys.param_env import env_get_int

from tools.tool_spec import (
    MAX_TOOL_PARAMS,
    ParamSpec,
    TOOL_COUNT,
    tool_specs,
)


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
        prop.__setitem__("type", value=gemini_type(p.dtype))
        prop.__setitem__("description", value=p.description)
        if p.has_minimum:
            prop.__setitem__("minimum", value=p.minimum)
        if p.has_items:
            var items = builtins.dict()
            items.__setitem__("type", value=gemini_type(p.items_type))
            prop.__setitem__("items", value=items)
        props.__setitem__(p.name, value=prop)
        if p.required:
            required.append(p.name)

    var schema = builtins.dict()
    schema.__setitem__("type", value="OBJECT")
    schema.__setitem__("properties", value=props)
    if builtins.len(required) > 0:
        schema.__setitem__("required", value=required)
    return schema


fn gemini_type(dtype: String) -> String:
    var lower = dtype.lower()
    if lower == "integer":
        return "INTEGER"
    if lower == "number":
        return "NUMBER"
    if lower == "boolean":
        return "BOOLEAN"
    if lower == "array":
        return "ARRAY"
    if lower == "object":
        return "OBJECT"
    return "STRING"


fn tools_gemini_object() raises -> PythonObject:
    var builtins = Python.import_module("builtins")
    var tools = builtins.list()
    var decls = builtins.list()
    var specs = tool_specs()
    for i in range(TOOL_COUNT):
        var spec = specs[i]
        var decl = builtins.dict()
        decl.__setitem__("name", value=spec.name)
        decl.__setitem__("description", value=spec.description)
        decl.__setitem__(
            "parameters",
            value=build_parameters_schema(spec.params, spec.param_count),
        )
        decls.append(decl)
    var tool = builtins.dict()
    tool.__setitem__("functionDeclarations", value=decls)
    tools.append(tool)
    return tools


fn build_gemini_messages() raises -> PythonObject:
    var builtins = Python.import_module("builtins")
    return builtins.list()


fn append_gemini_system_message(messages: PythonObject, text: String) raises:
    var builtins = Python.import_module("builtins")
    var msg = builtins.dict()
    msg.__setitem__("role", value="system")
    var parts = builtins.list()
    var part = builtins.dict()
    part.__setitem__("text", value=text)
    parts.append(part)
    msg.__setitem__("parts", value=parts)
    messages.append(msg)


fn append_gemini_user_message(messages: PythonObject, text: String) raises:
    var builtins = Python.import_module("builtins")
    var msg = builtins.dict()
    msg.__setitem__("role", value="user")
    var parts = builtins.list()
    var part = builtins.dict()
    part.__setitem__("text", value=text)
    parts.append(part)
    msg.__setitem__("parts", value=parts)
    messages.append(msg)


fn append_gemini_tool_message(
    messages: PythonObject,
    name: String,
    content: String,
) raises:
    var builtins = Python.import_module("builtins")
    var msg = builtins.dict()
    msg.__setitem__("role", value="user")
    var parts = builtins.list()
    var response = builtins.dict()
    var payload = builtins.dict()
    payload.__setitem__("result", value=content)
    response.__setitem__("name", value=name)
    response.__setitem__("response", value=payload)
    var part = builtins.dict()
    part.__setitem__("functionResponse", value=response)
    parts.append(part)
    msg.__setitem__("parts", value=parts)
    messages.append(msg)


fn extract_gemini_tool_calls(message: PythonObject) raises -> PythonObject:
    var builtins = Python.import_module("builtins")
    var parts = message.get("parts")
    if parts is None:
        return builtins.list()
    var out = builtins.list()
    var count = Int(builtins.len(parts))
    for i in range(count):
        var part = parts.__getitem__(i)
        var call = part.get("functionCall")
        if call is not None:
            out.append(call)
    return out


fn get_gemini_tool_call_name(call: PythonObject) raises -> String:
    return String(call.__getitem__("name"))


fn get_gemini_tool_call_args(call: PythonObject) raises -> PythonObject:
    return call.__getitem__("args")


fn get_gemini_tool_call_id(index: Int) -> String:
    return "gemini-" + String(index)


fn parse_gemini_args(args: PythonObject) raises -> List[Tuple[String, String]]:
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


struct GeminiProvider:
    var base_url: String
    var model: String
    var api_key: String

    fn __init__(out self, base_url: String, model: String, api_key: String):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key

    fn list_models(self) raises -> String:
        var json = Python.import_module("json")
        var urllib = Python.import_module("urllib.request")
        var builtins = Python.import_module("builtins")
        var os = Python.import_module("os")
        var os = Python.import_module("os")
        var os = Python.import_module("os")

        var url = self.base_url.rstrip("/") + "/models?key=" + self.api_key
        var py_globals = builtins.dict()
        py_globals.__setitem__("__builtins__", value=builtins.__dict__)
        var py_code =
            "import urllib.request, urllib.error\n"
            "def _gemini_get(url):\n"
            "    try:\n"
            "        with urllib.request.urlopen(url, timeout=60) as resp:\n"
            "            return {'ok': True, 'body': resp.read().decode('utf-8')}\n"
            "    except urllib.error.HTTPError as e:\n"
            "        return {'ok': False, 'status': e.code, 'body': e.read().decode('utf-8')}\n"
            "    except Exception as e:\n"
            "        return {'ok': False, 'error': repr(e)}\n"
        builtins.exec(py_code, py_globals)
        var helper = py_globals.__getitem__("_gemini_get")
        var result = helper(url)
        var ok = Bool(result.__getitem__("ok"))
        if not ok:
            if result.get("body") is not None:
                raise Error("HTTP error from Gemini: " + String(result.__getitem__("body")))
            if result.get("status") is not None:
                raise Error("HTTP error from Gemini (" + String(result.__getitem__("status")) + ")")
            if result.get("error") is not None:
                raise Error("HTTP error from Gemini: " + String(result.__getitem__("error")))
            raise Error("HTTP error from Gemini. Check GEMINI_API_KEY and base URL.")

        var response = json.loads(String(result.__getitem__("body")))
        var models = response.get("models")
        if models is None:
            return "No models returned."
        var count = Int(builtins.len(models))
        if count == 0:
            return "No models returned."

        var lines = List[String](capacity=count + 1)
        lines.append("Available Gemini models:")
        for i in range(count):
            var model = models.__getitem__(i)
            var name = String(model.__getitem__("name"))
            var methods = model.get("supportedGenerationMethods")
            var methods_text = ""
            if methods is not None:
                var method_list = builtins.list(methods)
                for j in range(Int(builtins.len(method_list))):
                    if j > 0:
                        methods_text += ", "
                    methods_text += String(method_list.__getitem__(j))
            if methods_text != "":
                lines.append("- " + name + " (methods: " + methods_text + ")")
            else:
                lines.append("- " + name)
        var out = ""
        for i in range(len(lines)):
            if i > 0:
                out += "\n"
            out += lines[i]
        return out

    fn chat(
        self,
        messages: PythonObject,
        tools: PythonObject,
        require_tools: Bool,
    ) raises -> PythonObject:
        var json = Python.import_module("json")
        var urllib = Python.import_module("urllib.request")
        var builtins = Python.import_module("builtins")

        var contents = builtins.list()
        var system_text = ""
        var count = Int(builtins.len(messages))
        for i in range(count):
            var msg = messages.__getitem__(i)
            var role = String(msg.__getitem__("role"))
            if role == "system":
                var parts = msg.__getitem__("parts")
                if parts is not None and builtins.len(parts) > 0:
                    var part = parts.__getitem__(0)
                    system_text = String(part.__getitem__("text"))
            else:
                contents.append(msg)

        var payload = builtins.dict()
        payload.__setitem__("contents", value=contents)
        payload.__setitem__("tools", value=tools)
        if system_text != "":
            var sys = builtins.dict()
            var parts = builtins.list()
            var part = builtins.dict()
            part.__setitem__("text", value=system_text)
            parts.append(part)
            sys.__setitem__("parts", value=parts)
            payload.__setitem__("systemInstruction", value=sys)
        var tool_config = builtins.dict()
        var fcfg = builtins.dict()
        fcfg.__setitem__("mode", value="AUTO")
        tool_config.__setitem__("functionCallingConfig", value=fcfg)
        payload.__setitem__("toolConfig", value=tool_config)

        var headers = builtins.dict()
        headers.__setitem__("Content-Type", value="application/json")
        headers.__setitem__("x-goog-api-key", value=self.api_key)

        var data = json.dumps(payload).encode("utf-8")
        var url = self.base_url.rstrip("/") + "/models/" + self.model + ":generateContent"
        var req = urllib.Request(
            url,
            data=data,
            headers=headers,
            method="POST",
        )
        if self.api_key == "":
            raise Error("GEMINI_API_KEY is not set.")
        comptime gemini_debug = (
            env_get_int["GEMINI_DEBUG"]() != 0
            if is_defined["GEMINI_DEBUG"]()
            else False
        )
        var debug = gemini_debug
        if debug:
            print("Gemini URL: " + url)
            print("Gemini payload: " + String(json.dumps(payload)))
        var raw = ""
        var py_globals = builtins.dict()
        py_globals.__setitem__("__builtins__", value=builtins.__dict__)
        var py_code =
            "import urllib.request, urllib.error\n"
            "def _gemini_post(url, headers, data):\n"
            "    req = urllib.request.Request(url, data=data, headers=headers, method='POST')\n"
            "    try:\n"
            "        with urllib.request.urlopen(req, timeout=60) as resp:\n"
            "            return {'ok': True, 'body': resp.read().decode('utf-8')}\n"
            "    except urllib.error.HTTPError as e:\n"
            "        return {'ok': False, 'status': e.code, 'body': e.read().decode('utf-8')}\n"
            "    except Exception as e:\n"
            "        return {'ok': False, 'error': repr(e)}\n"
        builtins.exec(py_code, py_globals)
        var helper = py_globals.__getitem__("_gemini_post")
        var result = helper(url, headers, data)
        var ok = Bool(result.__getitem__("ok"))
        if ok:
            raw = String(result.__getitem__("body"))
        else:
            if result.get("body") is not None:
                raise Error("HTTP error from Gemini: " + String(result.__getitem__("body")))
            if result.get("status") is not None:
                raise Error("HTTP error from Gemini (" + String(result.__getitem__("status")) + ")")
            if result.get("error") is not None:
                raise Error("HTTP error from Gemini: " + String(result.__getitem__("error")))
            raise Error("HTTP error from Gemini. Check GEMINI_API_KEY, model, and base URL.")
        var response = json.loads(raw)
        var candidates = response.__getitem__("candidates")
        var first = candidates.__getitem__(0)
        return first.__getitem__("content")
