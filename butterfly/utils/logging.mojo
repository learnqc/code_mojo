from butterfly.utils.config import Config
from butterfly.utils.benchmark_utils import get_timestamp_string

alias LOG_DEBUG = 10
alias LOG_INFO = 20
alias LOG_WARN = 30
alias LOG_ERROR = 40

# Compile-time log filter; raise this to drop lower-level logs at compile time.
alias COMPILE_LOG_LEVEL = LOG_INFO


fn _parse_log_level(level: String, default: Int) -> Int:
    var lower = level.lower()
    if lower == "debug":
        return LOG_DEBUG
    if lower == "info":
        return LOG_INFO
    if lower == "warn" or lower == "warning":
        return LOG_WARN
    if lower == "error":
        return LOG_ERROR
    try:
        return Int(level)
    except:
        return default


@parameter
fn _compile_enabled(level: Int) -> Bool:
    return COMPILE_LOG_LEVEL <= level


struct LoggingCtx(Movable):
    """Pure Mojo logging context."""

    var level: Int
    var prefix: String
    var show_time: Bool

    fn __init__(
        out self,
        level: Int = LOG_INFO,
        prefix: String = "",
        show_time: Bool = False,
    ):
        self.level = level
        self.prefix = prefix
        self.show_time = show_time

    fn format_prefix(self) -> String:
        var out = ""
        if self.show_time:
            out += "[" + get_timestamp_string() + "] "
        if self.prefix != "":
            out += "[" + self.prefix + "] "
        return out

    @staticmethod
    fn from_config(
        cfg: Config, default_level: Int = LOG_INFO, default_prefix: String = ""
    ) raises -> LoggingCtx:
        var level_str = cfg.get_value("log_level", "")
        var level = (
            _parse_log_level(level_str, default_level)
            if level_str != ""
            else default_level
        )
        var prefix = cfg.get_value("log_prefix", default_prefix)
        var show_time = cfg.get_bool("log_time", False)
        return LoggingCtx(level, prefix, show_time)


fn default_log_ctx() raises -> LoggingCtx:
    return LoggingCtx.from_config(Config.load("butterfly.conf"))


fn log_debug(msg: String, ctx: LoggingCtx = LoggingCtx()):
    if not _compile_enabled(LOG_DEBUG):
        return
    if ctx.level <= LOG_DEBUG:
        print("[DEBUG] " + ctx.format_prefix() + msg)


fn log_info(msg: String, ctx: LoggingCtx = LoggingCtx()):
    if not _compile_enabled(LOG_INFO):
        return
    if ctx.level <= LOG_INFO:
        print("[INFO] " + ctx.format_prefix() + msg)


fn log_warn(msg: String, ctx: LoggingCtx = LoggingCtx()):
    if not _compile_enabled(LOG_WARN):
        return
    if ctx.level <= LOG_WARN:
        print("[WARN] " + ctx.format_prefix() + msg)


fn log_error(msg: String, ctx: LoggingCtx = LoggingCtx()):
    if not _compile_enabled(LOG_ERROR):
        return
    if ctx.level <= LOG_ERROR:
        print("[ERROR] " + ctx.format_prefix() + msg)
