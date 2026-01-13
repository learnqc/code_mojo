# Config

There are two supported patterns:

1) Pure Mojo config object (explicit dependency)
   - Load once and pass through your call chain.
   - No global state, easiest to test.

```mojo
from butterfly.utils.config import Config
from butterfly.utils.context import ExecContext

var cfg = Config.load("butterfly.conf")
var ctx = ExecContext.from_config(cfg)
```

2) Python-backed global config (cached)
   - Convenience API; uses Python interop and a process-wide cache.
   - Useful when you want a shared config without threading it through.

```mojo
from butterfly.utils.context import ExecContext

var ctx = ExecContext.from_global_config()
```

Config file format:

```
threads=8
```

Notes:
- `threads <= 0` means "use runtime default."
- The global config reads `BUTTERFLY_CONFIG_PATH` if set.
- For pure Mojo builds (no Python), use `LoggingCtx` and `log_*` helpers.
