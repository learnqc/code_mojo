from python import Python, PythonObject
from butterfly.core.types import FloatType

fn _get_state() raises -> PythonObject:
    var sys = Python.import_module("sys")
    var types = Python.import_module("types")
    var modules = sys.modules
    var name = "butterfly_global_config_state"
    try:
        return modules.__getitem__(name)
    except:
        var state = types.SimpleNamespace()
        state.loaded = False
        state.path = "butterfly.conf"
        state.cache = {}
        state.load_count = 0
        modules.__setitem__(name, value=state)
        return state


fn set_global_config_path(path: String) raises:
    """Set config path and reset cache."""
    var state = _get_state()
    state.path = path
    state.loaded = False
    state.cache = {}
    state.load_count = 0


fn reload_global_config() raises:
    """Force reload of the config file."""
    var state = _get_state()
    state.loaded = False
    state.cache = {}
    state.load_count = 0
    load_global_config("")


fn load_global_config(path: String = "") raises:
    """Load config into cache (key=value per line)."""
    var state = _get_state()
    if state.loaded:
        return

    var final_path = path
    if final_path == "":
        var os = Python.import_module("os")
        var env_path = os.getenv("BUTTERFLY_CONFIG_PATH")
        if env_path is None:
            final_path = String(state.path)
        else:
            final_path = String(env_path)

    try:
        with open(final_path, "r") as f:
            var content = f.read()
            var lines = content.split("\n")
            for i in range(len(lines)):
                var line = String(lines[i]).strip()
                if line == "" or line.startswith("#") or line.startswith(";"):
                    continue
                var idx = line.find("=")
                if idx < 0:
                    continue
                var key_str = String(line[:idx].strip())
                var val_str = String(line[idx + 1 :].strip())
                if len(key_str) > 0:
                    state.cache.__setitem__(key_str, value=val_str)
    except:
        # Missing or unreadable config: treat as empty.
        pass
    state.load_count = state.load_count + 1
    state.loaded = True


fn get_global_config_value(key: String, default: String = "") raises -> String:
    """Read a config value with caching."""
    var state = _get_state()
    if not state.loaded:
        load_global_config("")
    var cached = state.cache.get(key)
    if cached is None:
        return default
    return String(cached)


fn get_global_config_load_count() raises -> Int:
    """Return how many times the config file has been loaded."""
    var state = _get_state()
    return Int(state.load_count)


fn get_global_config_int(key: String, default: Int = 0) raises -> Int:
    var val = get_global_config_value(key, "")
    if val == "":
        return default
    try:
        return Int(val)
    except:
        return default


fn get_global_config_float(
    key: String, default: FloatType = 0.0
) raises -> FloatType:
    var val = get_global_config_value(key, "")
    if val == "":
        return default
    try:
        return FloatType(Float64(val))
    except:
        return default


fn get_global_config_bool(
    key: String, default: Bool = False
) raises -> Bool:
    var val = get_global_config_value(key, "")
    if val == "":
        return default
    var lower = val.lower()
    if lower == "true" or lower == "1" or lower == "yes" or lower == "on":
        return True
    if lower == "false" or lower == "0" or lower == "no" or lower == "off":
        return False
    return default
