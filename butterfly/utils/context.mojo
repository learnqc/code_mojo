from butterfly.utils.config import Config
from butterfly.utils.config_global import get_global_config_bool, get_global_config_int

alias SIMD_USE_SPECIALIZED_H_DEFAULT = True
alias SIMD_USE_SPECIALIZED_P_DEFAULT = True
alias SIMD_USE_SPECIALIZED_CP_DEFAULT = True
alias SIMD_USE_SPECIALIZED_X_DEFAULT = True
alias SIMD_USE_SPECIALIZED_CX_DEFAULT = True
alias SIMD_USE_SPECIALIZED_RY_DEFAULT = True
alias SIMD_USE_SPECIALIZED_CRY_DEFAULT = True

struct ExecutionStrategy(Copyable, Movable, ImplicitlyCopyable):
    var value: Int

    fn __init__(out self, value: Int):
        self.value = value

    fn __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    fn __ne__(self, other: Self) -> Bool:
        return self.value != other.value

    fn __str__(self) -> String:
        return execution_strategy_name(self)

    @staticmethod
    fn from_int(value: Int) raises -> ExecutionStrategy:
        if not is_valid_execution_strategy(value):
            raise Error("Unknown execution strategy: " + String(value))
        return ExecutionStrategy(value)

    alias SCALAR = ExecutionStrategy(0)
    alias SCALAR_PARALLEL = ExecutionStrategy(1)
    alias SIMD = ExecutionStrategy(2)
    alias SIMD_PARALLEL = ExecutionStrategy(3)
    alias GRID = ExecutionStrategy(4)
    alias GRID_PARALLEL = ExecutionStrategy(5)
    alias GRID_FUSED = ExecutionStrategy(6)
    alias GRID_PARALLEL_FUSED = ExecutionStrategy(7)


fn is_valid_execution_strategy(value: Int) -> Bool:
    return value >= ExecutionStrategy.SCALAR.value and value <= ExecutionStrategy.GRID_PARALLEL_FUSED.value


fn execution_strategy_name(strategy: ExecutionStrategy) -> String:
    if strategy == ExecutionStrategy.SCALAR:
        return "SCALAR"
    if strategy == ExecutionStrategy.SCALAR_PARALLEL:
        return "SCALAR_PARALLEL"
    if strategy == ExecutionStrategy.SIMD:
        return "SIMD"
    if strategy == ExecutionStrategy.SIMD_PARALLEL:
        return "SIMD_PARALLEL"
    if strategy == ExecutionStrategy.GRID:
        return "GRID"
    if strategy == ExecutionStrategy.GRID_PARALLEL:
        return "GRID_PARALLEL"
    if strategy == ExecutionStrategy.GRID_FUSED:
        return "GRID_FUSED"
    if strategy == ExecutionStrategy.GRID_PARALLEL_FUSED:
        return "GRID_PARALLEL_FUSED"
    return "UNKNOWN"


struct ExecContext(Copyable, Movable):
    """Execution context for runtime tuning parameters."""

    var threads: Int
    var quantum_simd_parallel_chunks: Int
    var parallel_flat: Bool
    var auto_parallel_log_n_min: Int
    var auto_parallel_scalar_target_max: Int
    var auto_simd_target_slack: Int
    var simd_min_stride_w4: Int
    var simd_min_stride_w8: Int
    var simd_min_stride_w16: Int
    var simd_use_specialized_h: Bool
    var simd_use_specialized_p: Bool
    var simd_use_specialized_cp: Bool
    var simd_use_specialized_x: Bool
    var simd_use_specialized_cx: Bool
    var simd_use_specialized_ry: Bool
    var simd_use_specialized_cry: Bool
    var grid_use_parallel: Bool
    var grid_col_bits_min: Int
    var grid_col_bits_slack: Int
    var validate_circuit: Bool
    var execution_strategy: ExecutionStrategy
    
    fn __init__(out self):
        # threads < 0 means "force sequential"
        # threads == 0 means "use runtime default"
        self.threads = 0
        self.quantum_simd_parallel_chunks = 0
        self.parallel_flat = False
        # Auto-selection thresholds.
        self.auto_parallel_log_n_min = 16
        self.auto_parallel_scalar_target_max = 0
        self.auto_simd_target_slack = 2
        self.simd_min_stride_w4 = 4
        self.simd_min_stride_w8 = 8
        self.simd_min_stride_w16 = 16
        self.simd_use_specialized_h = SIMD_USE_SPECIALIZED_H_DEFAULT
        self.simd_use_specialized_p = SIMD_USE_SPECIALIZED_P_DEFAULT
        self.simd_use_specialized_cp = SIMD_USE_SPECIALIZED_CP_DEFAULT
        self.simd_use_specialized_x = SIMD_USE_SPECIALIZED_X_DEFAULT
        self.simd_use_specialized_cx = SIMD_USE_SPECIALIZED_CX_DEFAULT
        self.simd_use_specialized_ry = SIMD_USE_SPECIALIZED_RY_DEFAULT
        self.simd_use_specialized_cry = SIMD_USE_SPECIALIZED_CRY_DEFAULT
        self.grid_use_parallel = True
        self.grid_col_bits_min = 3
        self.grid_col_bits_slack = 3
        self.validate_circuit = True
        self.execution_strategy = ExecutionStrategy.SIMD

    @staticmethod
    fn from_config(cfg: Config) raises -> ExecContext:
        var threads = cfg.get_int("threads", -1)
        ctx = ExecContext()
        ctx.threads = threads
        ctx.quantum_simd_parallel_chunks = cfg.get_int(
            "quantum_simd_parallel_chunks",
            ctx.quantum_simd_parallel_chunks,
        )
        ctx.parallel_flat = cfg.get_bool(
            "parallel_flat",
            ctx.parallel_flat,
        )
        ctx.auto_parallel_log_n_min = cfg.get_int(
            "auto_parallel_log_n_min",
            ctx.auto_parallel_log_n_min,
        )
        ctx.auto_parallel_scalar_target_max = cfg.get_int(
            "auto_parallel_scalar_target_max",
            ctx.auto_parallel_scalar_target_max,
        )
        ctx.auto_simd_target_slack = cfg.get_int(
            "auto_simd_target_slack",
            ctx.auto_simd_target_slack,
        )
        ctx.simd_min_stride_w4 = cfg.get_int(
            "simd_min_stride_w4",
            ctx.simd_min_stride_w4,
        )
        ctx.simd_min_stride_w8 = cfg.get_int(
            "simd_min_stride_w8",
            ctx.simd_min_stride_w8,
        )
        ctx.simd_min_stride_w16 = cfg.get_int(
            "simd_min_stride_w16",
            ctx.simd_min_stride_w16,
        )
        ctx.simd_use_specialized_h = cfg.get_bool(
            "simd_use_specialized_h",
            ctx.simd_use_specialized_h,
        )
        ctx.simd_use_specialized_p = cfg.get_bool(
            "simd_use_specialized_p",
            ctx.simd_use_specialized_p,
        )
        ctx.simd_use_specialized_cp = cfg.get_bool(
            "simd_use_specialized_cp",
            ctx.simd_use_specialized_cp,
        )
        ctx.simd_use_specialized_x = cfg.get_bool(
            "simd_use_specialized_x",
            ctx.simd_use_specialized_x,
        )
        ctx.simd_use_specialized_cx = cfg.get_bool(
            "simd_use_specialized_cx",
            ctx.simd_use_specialized_cx,
        )
        ctx.simd_use_specialized_ry = cfg.get_bool(
            "simd_use_specialized_ry",
            ctx.simd_use_specialized_ry,
        )
        ctx.simd_use_specialized_cry = cfg.get_bool(
            "simd_use_specialized_cry",
            ctx.simd_use_specialized_cry,
        )
        ctx.grid_use_parallel = cfg.get_bool(
            "grid_use_parallel",
            ctx.grid_use_parallel,
        )
        ctx.grid_col_bits_min = cfg.get_int(
            "grid_col_bits_min",
            ctx.grid_col_bits_min,
        )
        ctx.grid_col_bits_slack = cfg.get_int(
            "grid_col_bits_slack",
            ctx.grid_col_bits_slack,
        )
        ctx.validate_circuit = cfg.get_bool(
            "validate_circuit",
            ctx.validate_circuit,
        )
        var strategy_value = cfg.get_int(
            "execution_strategy",
            ctx.execution_strategy.value,
        )
        ctx.execution_strategy = ExecutionStrategy.from_int(strategy_value)
        return ctx^

    @staticmethod
    fn from_global_config(default_threads: Int = 0) raises -> ExecContext:
        var threads = get_global_config_int("threads", default_threads)
        ctx = ExecContext()
        ctx.threads = threads
        ctx.quantum_simd_parallel_chunks = get_global_config_int(
            "quantum_simd_parallel_chunks",
            ctx.quantum_simd_parallel_chunks,
        )
        ctx.parallel_flat = get_global_config_bool(
            "parallel_flat",
            ctx.parallel_flat,
        )
        ctx.auto_parallel_log_n_min = get_global_config_int(
            "auto_parallel_log_n_min",
            ctx.auto_parallel_log_n_min,
        )
        ctx.auto_parallel_scalar_target_max = get_global_config_int(
            "auto_parallel_scalar_target_max",
            ctx.auto_parallel_scalar_target_max,
        )
        ctx.auto_simd_target_slack = get_global_config_int(
            "auto_simd_target_slack",
            ctx.auto_simd_target_slack,
        )
        ctx.simd_min_stride_w4 = get_global_config_int(
            "simd_min_stride_w4",
            ctx.simd_min_stride_w4,
        )
        ctx.simd_min_stride_w8 = get_global_config_int(
            "simd_min_stride_w8",
            ctx.simd_min_stride_w8,
        )
        ctx.simd_min_stride_w16 = get_global_config_int(
            "simd_min_stride_w16",
            ctx.simd_min_stride_w16,
        )
        ctx.simd_use_specialized_h = get_global_config_bool(
            "simd_use_specialized_h",
            ctx.simd_use_specialized_h,
        )
        ctx.simd_use_specialized_p = get_global_config_bool(
            "simd_use_specialized_p",
            ctx.simd_use_specialized_p,
        )
        ctx.simd_use_specialized_cp = get_global_config_bool(
            "simd_use_specialized_cp",
            ctx.simd_use_specialized_cp,
        )
        ctx.simd_use_specialized_x = get_global_config_bool(
            "simd_use_specialized_x",
            ctx.simd_use_specialized_x,
        )
        ctx.simd_use_specialized_cx = get_global_config_bool(
            "simd_use_specialized_cx",
            ctx.simd_use_specialized_cx,
        )
        ctx.simd_use_specialized_ry = get_global_config_bool(
            "simd_use_specialized_ry",
            ctx.simd_use_specialized_ry,
        )
        ctx.simd_use_specialized_cry = get_global_config_bool(
            "simd_use_specialized_cry",
            ctx.simd_use_specialized_cry,
        )
        ctx.grid_use_parallel = get_global_config_bool(
            "grid_use_parallel",
            ctx.grid_use_parallel,
        )
        ctx.grid_col_bits_min = get_global_config_int(
            "grid_col_bits_min",
            ctx.grid_col_bits_min,
        )
        ctx.grid_col_bits_slack = get_global_config_int(
            "grid_col_bits_slack",
            ctx.grid_col_bits_slack,
        )
        ctx.validate_circuit = get_global_config_bool(
            "validate_circuit",
            ctx.validate_circuit,
        )
        var strategy_value = get_global_config_int(
            "execution_strategy",
            ctx.execution_strategy.value,
        )
        ctx.execution_strategy = ExecutionStrategy.from_int(strategy_value)
        return ctx^
