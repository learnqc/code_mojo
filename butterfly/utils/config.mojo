from collections import Dict


struct Config(Movable):
    """Simple config loader with in-memory cache."""

    var path: String
    var cache: Dict[String, String]
    var load_count: Int

    fn __init__(out self, path: String):
        self.path = path
        self.cache = Dict[String, String]()
        self.load_count = 0

    @staticmethod
    fn load(
        path: String = "",
        env_var: String = "BUTTERFLY_CONFIG_PATH",
        default_path: String = "butterfly.conf",
    ) -> Config:
        from os import getenv

        var final_path = path
        if final_path == "":
            final_path = getenv(env_var, default_path)
        var cfg = Config(final_path)
        cfg.reload()
        return cfg^

    fn reload(mut self):
        self.cache = Dict[String, String]()
        self.load_count += 1
        try:
            with open(self.path, "r") as f:
                var content = f.read()
                var lines = content.split("\n")
                for i in range(len(lines)):
                    var line = String(lines[i]).strip()
                    if (
                        line == ""
                        or line.startswith("#")
                        or line.startswith(";")
                    ):
                        continue
                    var idx = line.find("=")
                    if idx < 0:
                        continue
                    var key_str = String(line[:idx].strip())
                    var val_str = String(line[idx + 1 :].strip())
                    if len(key_str) > 0:
                        self.cache[key_str] = val_str
        except:
            # Missing or unreadable config: treat as empty.
            pass

    fn get_value(self, key: String, default: String = "") raises -> String:
        var key_str = String(key)
        try:
            if key_str in self.cache:
                return self.cache[key_str]
            return default
        except:
            return default

    fn get_int(self, key: String, default: Int = 0) raises -> Int:
        var val = self.get_value(key, "")
        if val == "":
            return default
        try:
            return Int(val)
        except:
            return default

    fn get_float(self, key: String, default: Float64 = 0.0) raises -> Float64:
        var val = self.get_value(key, "")
        if val == "":
            return default
        try:
            return Float64(val)
        except:
            return default

    fn get_bool(self, key: String, default: Bool = False) raises -> Bool:
        var val = self.get_value(key, "")
        if val == "":
            return default
        var lower = val.lower()
        if lower == "true" or lower == "1" or lower == "yes" or lower == "on":
            return True
        if lower == "false" or lower == "0" or lower == "no" or lower == "off":
            return False
        return default
