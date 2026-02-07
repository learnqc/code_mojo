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
validate_circuit=true
grid_col_bits_min=3
grid_col_bits_slack=3
execution_strategy=2
```

Notes:
- `threads <= 0` means "use runtime default."
- `validate_circuit` toggles upfront circuit validation before execution.
- `grid_col_bits_min` sets the minimum number of column bits for grid execution.
- `grid_col_bits_slack` sets the slack for the heuristic `col_bits = max(n - slack, min)`.
- `execution_strategy` uses numeric values: 0=SCALAR, 1=SCALAR_PARALLEL, 2=SIMD, 3=SIMD_PARALLEL, 4=GRID, 5=GRID_PARALLEL, 6=GRID_FUSED, 7=GRID_PARALLEL_FUSED.
- The global config reads `BUTTERFLY_CONFIG_PATH` if set.
- For pure Mojo builds (no Python), use `LoggingCtx` and `log_*` helpers.
