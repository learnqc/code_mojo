"""
Butterfly Configuration System

Provides file-based configuration for parallelization, performance tuning,
algorithm selection, memory management, and debugging.
"""

from sys.info import num_physical_cores


# Config key aliases - centralized for maintainability
alias CFG_PARALLEL_DEFAULT = "parallelization.default"
alias CFG_PARALLEL_QUANTUM = "parallelization.quantum_workers"
alias CFG_PARALLEL_FOUR_QUARTERS = "parallelization.quantum_four_quarters"
alias CFG_PARALLEL_SIMD_V2_CHUNKS = "parallelization.quantum_simd_v2_chunks"
alias CFG_PARALLEL_FUSED_V3_LOCAL = "parallelization.fused_v3_local_blocks"
alias CFG_PARALLEL_FUSED_V3_RADIX4 = "parallelization.fused_v3_radix4_chunks"
alias CFG_PARALLEL_FUSED_V3_LOCAL_GRAIN = "parallelization.fused_v3_local_grain"
alias CFG_PARALLEL_FUSED_V3_RADIX4_GRAIN = "parallelization.fused_v3_radix4_grain"
alias CFG_PARALLEL_V_GRID_ROWS = "parallelization.v_grid_row_workers"
alias CFG_PARALLEL_V_GRID_COLS = "parallelization.v_grid_column_workers"
alias CFG_PARALLEL_FFT = "parallelization.fft_workers"
alias CFG_PARALLEL_FFT_GLOBAL = "parallelization.fft_global_stage"
alias CFG_PARALLEL_FFT_LOCAL = "parallelization.fft_local_stage"

alias CFG_PERF_SIMD_WIDTH = "performance.simd_width"
alias CFG_PERF_TILE_SIZE = "performance.tile_size"
alias CFG_PERF_L2_TILE_SIZE = "performance.l2_tile_size"
alias CFG_PERF_ALIGNMENT = "performance.alignment"

alias CFG_ALGO_STRATEGY = "algorithm.default_strategy"
alias CFG_ALGO_FFT = "algorithm.fft_algorithm"
alias CFG_ALGO_VERIFY = "algorithm.verification_enabled"
alias CFG_ALGO_CHECKS = "algorithm.correctness_checks"

alias CFG_MEM_BUFFER_POOL = "memory.buffer_pool_size"
alias CFG_MEM_PREALLOCATE = "memory.preallocate_buffers"
alias CFG_MEM_MAX_STATE = "memory.max_state_size"

alias CFG_DEBUG_VERBOSITY = "debugging.verbosity"
alias CFG_DEBUG_TIMING = "debugging.timing_enabled"
alias CFG_DEBUG_PROFILE = "debugging.profile_enabled"
alias CFG_DEBUG_LOG_FILE = "debugging.log_file"


fn detect_physical_cores() -> Int:
    """Detect number of physical CPU cores, fallback to 16 if detection fails.
    """
    var cores = num_physical_cores()
    if cores > 0:
        return cores
    return 8  # Fallback


struct ButterflyConfig(Copyable):
    """Comprehensive Butterfly configuration."""

    # Parallelization
    var parallel_default: Int
    var parallel_quantum: Int
    var parallel_quantum_four_quarters: Int
    var parallel_quantum_simd_v2_chunks: Int  # Worker chunks for SIMD v2 kernels
    var parallel_fused_v3_local_blocks: Int  # Cache block workers for fused_v3
    var parallel_fused_v3_radix4_chunks: Int  # Radix-4 workers for fused_v3
    var parallel_fused_v3_local_grain: Int  # Blocks per work item (original: 1)
    var parallel_fused_v3_radix4_grain: Int  # Butterflies per work item (original: 1)
    var parallel_fft: Int
    var parallel_fft_global: Int
    var parallel_fft_local: Int
    var parallel_v_grid_rows: Int
    var parallel_v_grid_columns: Int

    # Performance
    var simd_width: Int
    var tile_size: Int
    var l2_tile_size: Int
    var alignment: Int

    # Algorithm
    var default_strategy: String
    var fft_algorithm: String
    var verification_enabled: Bool
    var correctness_checks: Bool

    # Memory
    var buffer_pool_size: Int
    var preallocate_buffers: Bool
    var max_state_size: Int

    # Debugging
    var verbosity: Int
    var timing_enabled: Bool
    var profile_enabled: Bool
    var log_file: String

    fn __init__(out self):
        """Initialize with safe defaults."""
        # Parallelization defaults
        self.parallel_default = 0
        self.parallel_quantum = 0
        self.parallel_quantum_four_quarters = 4
        self.parallel_quantum_simd_v2_chunks = (
            0  # 0 = auto-detect physical cores
        )
        self.parallel_fused_v3_local_blocks = 0  # 0 = auto-detect
        self.parallel_fused_v3_radix4_chunks = 0  # 0 = auto-detect
        self.parallel_fused_v3_local_grain = (
            1  # Original: 1 block per work item
        )
        self.parallel_fused_v3_radix4_grain = (
            1  # Original: 1 butterfly per work item
        )
        self.parallel_fft = 0
        self.parallel_fft_global = 0
        self.parallel_fft_local = 0
        self.parallel_v_grid_rows = 0  # 0 = unlimited
        self.parallel_v_grid_columns = 0  # 0 = unlimited

        # Performance defaults
        self.simd_width = 0  # Auto-detect
        self.tile_size = 32768  # 32KB L1 cache
        self.l2_tile_size = 4194304  # 4MB L2 cache
        self.alignment = 64  # Cache line

        # Algorithm defaults
        self.default_strategy = "auto"
        self.fft_algorithm = "phast"
        self.verification_enabled = True
        self.correctness_checks = True

        # Memory defaults
        self.buffer_pool_size = 0
        self.preallocate_buffers = False
        self.max_state_size = 0

        # Debugging defaults
        self.verbosity = 0
        self.timing_enabled = False
        self.profile_enabled = False
        self.log_file = ""

    fn __copyinit__(out self, existing: Self):
        """Copy constructor."""
        self.parallel_default = existing.parallel_default
        self.parallel_quantum = existing.parallel_quantum
        self.parallel_quantum_four_quarters = (
            existing.parallel_quantum_four_quarters
        )
        self.parallel_quantum_simd_v2_chunks = (
            existing.parallel_quantum_simd_v2_chunks
        )
        self.parallel_fused_v3_local_blocks = (
            existing.parallel_fused_v3_local_blocks
        )
        self.parallel_fused_v3_radix4_chunks = (
            existing.parallel_fused_v3_radix4_chunks
        )
        self.parallel_fused_v3_local_grain = (
            existing.parallel_fused_v3_local_grain
        )
        self.parallel_fused_v3_radix4_grain = (
            existing.parallel_fused_v3_radix4_grain
        )
        self.parallel_fft = existing.parallel_fft
        self.parallel_fft_global = existing.parallel_fft_global
        self.parallel_fft_local = existing.parallel_fft_local
        self.parallel_v_grid_rows = existing.parallel_v_grid_rows
        self.parallel_v_grid_columns = existing.parallel_v_grid_columns

        self.simd_width = existing.simd_width
        self.tile_size = existing.tile_size
        self.l2_tile_size = existing.l2_tile_size
        self.alignment = existing.alignment

        self.default_strategy = existing.default_strategy
        self.fft_algorithm = existing.fft_algorithm
        self.verification_enabled = existing.verification_enabled
        self.correctness_checks = existing.correctness_checks

        self.buffer_pool_size = existing.buffer_pool_size
        self.preallocate_buffers = existing.preallocate_buffers
        self.max_state_size = existing.max_state_size

        self.verbosity = existing.verbosity
        self.timing_enabled = existing.timing_enabled
        self.profile_enabled = existing.profile_enabled
        self.log_file = existing.log_file

    fn get_workers(self, operation_type: String) -> Int:
        """Get worker count for a specific operation type."""
        if operation_type == "quantum":
            return (
                self.parallel_quantum if self.parallel_quantum
                > 0 else self.parallel_default
            )
        elif operation_type == "quantum_four_quarters":
            return self.parallel_quantum_four_quarters
        elif operation_type == "quantum_simd_v2_chunks":
            if self.parallel_quantum_simd_v2_chunks > 0:
                return self.parallel_quantum_simd_v2_chunks
            else:
                # Auto-detect physical cores
                return detect_physical_cores()
        elif operation_type == "fused_v3_local_blocks":
            # 0 = unlimited (don't pass workers param), >0 = explicit limit
            return self.parallel_fused_v3_local_blocks
        elif operation_type == "fused_v3_radix4_chunks":
            # 0 = unlimited (don't pass workers param), >0 = explicit limit
            return self.parallel_fused_v3_radix4_chunks
        elif operation_type == "fft":
            return (
                self.parallel_fft if self.parallel_fft
                > 0 else self.parallel_default
            )
        elif operation_type == "fft_global":
            return (
                self.parallel_fft_global if self.parallel_fft_global
                > 0 else self.parallel_fft
            )
        elif operation_type == "fft_local":
            return (
                self.parallel_fft_local if self.parallel_fft_local
                > 0 else self.parallel_fft
            )
        elif operation_type == "v_grid_rows":
            return self.parallel_v_grid_rows
        elif operation_type == "v_grid_columns":
            return self.parallel_v_grid_columns
        return self.parallel_default

    @staticmethod
    fn from_file(path: String) raises -> ButterflyConfig:
        """Load configuration from a simple key-value file."""
        var config = ButterflyConfig()

        # Read file content
        with open(path, "r") as f:
            var content = f.read()
            var lines = content.split("\n")

            for i in range(len(lines)):
                var line = lines[i].strip()

                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                # Parse key=value
                var parts = line.split("=")
                if len(parts) != 2:
                    continue

                var key = parts[0].strip()
                var value = parts[1].strip()

                # Parse parallelization settings
                if key == CFG_PARALLEL_DEFAULT:
                    config.parallel_default = atol(value)
                elif key == CFG_PARALLEL_QUANTUM:
                    config.parallel_quantum = atol(value)
                elif key == CFG_PARALLEL_FOUR_QUARTERS:
                    config.parallel_quantum_four_quarters = atol(value)
                elif key == CFG_PARALLEL_SIMD_V2_CHUNKS:
                    config.parallel_quantum_simd_v2_chunks = atol(value)
                elif key == CFG_PARALLEL_FUSED_V3_LOCAL:
                    config.parallel_fused_v3_local_blocks = atol(value)
                elif key == CFG_PARALLEL_FUSED_V3_RADIX4:
                    config.parallel_fused_v3_radix4_chunks = atol(value)
                elif key == CFG_PARALLEL_FUSED_V3_LOCAL_GRAIN:
                    config.parallel_fused_v3_local_grain = atol(value)
                elif key == CFG_PARALLEL_FUSED_V3_RADIX4_GRAIN:
                    config.parallel_fused_v3_radix4_grain = atol(value)
                elif key == CFG_PARALLEL_V_GRID_ROWS:
                    config.parallel_v_grid_rows = atol(value)
                elif key == CFG_PARALLEL_V_GRID_COLS:
                    config.parallel_v_grid_columns = atol(value)
                elif key == CFG_PARALLEL_FFT:
                    config.parallel_fft = atol(value)
                elif key == CFG_PARALLEL_FFT_GLOBAL:
                    config.parallel_fft_global = atol(value)
                elif key == CFG_PARALLEL_FFT_LOCAL:
                    config.parallel_fft_local = atol(value)

                # Parse performance settings
                elif key == CFG_PERF_SIMD_WIDTH:
                    config.simd_width = atol(value)
                elif key == CFG_PERF_TILE_SIZE:
                    config.tile_size = atol(value)
                elif key == CFG_PERF_L2_TILE_SIZE:
                    config.l2_tile_size = atol(value)
                elif key == CFG_PERF_ALIGNMENT:
                    config.alignment = atol(value)

                # Parse algorithm settings
                elif key == CFG_ALGO_STRATEGY:
                    config.default_strategy = String(value)
                elif key == CFG_ALGO_FFT:
                    config.fft_algorithm = String(value)
                elif key == CFG_ALGO_VERIFY:
                    config.verification_enabled = value.lower() == "true"
                elif key == CFG_ALGO_CHECKS:
                    config.correctness_checks = value.lower() == "true"

                # Parse memory settings
                elif key == CFG_MEM_BUFFER_POOL:
                    config.buffer_pool_size = atol(value)
                elif key == CFG_MEM_PREALLOCATE:
                    config.preallocate_buffers = value.lower() == "true"
                elif key == CFG_MEM_MAX_STATE:
                    config.max_state_size = atol(value)

                # Parse debugging settings
                elif key == CFG_DEBUG_VERBOSITY:
                    config.verbosity = atol(value)
                elif key == CFG_DEBUG_TIMING:
                    config.timing_enabled = value.lower() == "true"
                elif key == CFG_DEBUG_PROFILE:
                    config.profile_enabled = value.lower() == "true"
                elif key == CFG_DEBUG_LOG_FILE:
                    config.log_file = String(value)

        return config.copy()

    fn print_config(self):
        """Print current configuration."""
        print("=== Butterfly Configuration ===")
        print("\n[Parallelization]")
        print("  default:", self.parallel_default)
        print("  quantum_workers:", self.parallel_quantum)
        print("  quantum_four_quarters:", self.parallel_quantum_four_quarters)
        print(
            "  quantum_simd_v2_chunks:",
            self.parallel_quantum_simd_v2_chunks,
            "(0 = auto-detect)",
        )
        print("  fft_workers:", self.parallel_fft)
        print("  fft_global_stage:", self.parallel_fft_global)
        print("  fft_local_stage:", self.parallel_fft_local)

        print("\n[Performance]")
        print("  simd_width:", self.simd_width)
        print("  tile_size:", self.tile_size)
        print("  l2_tile_size:", self.l2_tile_size)
        print("  alignment:", self.alignment)

        print("\n[Algorithm]")
        print("  default_strategy:", self.default_strategy)
        print("  fft_algorithm:", self.fft_algorithm)
        print("  verification_enabled:", self.verification_enabled)
        print("  correctness_checks:", self.correctness_checks)

        print("\n[Memory]")
        print("  buffer_pool_size:", self.buffer_pool_size)
        print("  preallocate_buffers:", self.preallocate_buffers)
        print("  max_state_size:", self.max_state_size)

        print("\n[Debugging]")
        print("  verbosity:", self.verbosity)
        print("  timing_enabled:", self.timing_enabled)
        print("  profile_enabled:", self.profile_enabled)
        print("  log_file:", self.log_file if self.log_file else "(none)")


fn load_config() -> ButterflyConfig:
    """Load configuration from file with fallback chain.

    Checks in order:
    1. BUTTERFLY_CONFIG_PATH environment variable
    2. butterfly.config in current directory
    3. Default configuration
    """
    from os import getenv

    # Try environment variable first
    var config_path = getenv("BUTTERFLY_CONFIG_PATH", "")
    if config_path:
        try:
            return ButterflyConfig.from_file(config_path)
        except:
            print(
                "Warning: Failed to load config from BUTTERFLY_CONFIG_PATH:",
                config_path,
            )

    # Try current directory
    try:
        return ButterflyConfig.from_file("butterfly.config")
    except:
        pass

    # Use defaults
    return ButterflyConfig()


# Convenience function to get config
fn get_config() -> ButterflyConfig:
    """Get configuration (loads fresh each time for simplicity)."""
    return load_config()


fn load_config_from_path(path: String) raises -> ButterflyConfig:
    """Load configuration from a specific path.

    Useful for tests and benchmarks that need to switch configs programmatically.

    Example:
        var config = load_config_from_path("configs/workers_2.config")
        var workers = config.get_workers("v_grid_rows")
    """
    return ButterflyConfig.from_file(path)


fn get_workers(operation_type: String) -> Int:
    """Get worker count for a specific operation type."""
    return get_config().get_workers(operation_type)


fn get_workers_from_config(
    config: ButterflyConfig, operation_type: String
) -> Int:
    """Get worker count for a specific operation type from a given config.

    Useful when you've already loaded a config and want to query it.

    Example:
        var config = load_config_from_path("configs/workers_8.config")
        var workers = get_workers_from_config(config, "v_grid_rows")
    """
    return config.get_workers(operation_type)


fn is_timing_enabled() -> Bool:
    """Check if timing is enabled."""
    return get_config().timing_enabled


fn get_verbosity() -> Int:
    """Get debug verbosity level."""
    return get_config().verbosity


fn get_default_strategy() -> String:
    """Get the default execution strategy."""
    return get_config().default_strategy


fn create_test_config(
    v_grid_row_workers: Int = 0,
    v_grid_column_workers: Int = 0,
    output_path: String = "butterfly.config",
) raises:
    """Create a minimal config file with specific worker counts.

    Useful for tests that need to programmatically set worker counts.

    Args:
        v_grid_row_workers: Worker count for row operations (0 = unlimited)
        v_grid_column_workers: Worker count for column operations (0 = unlimited)
        output_path: Where to write the config (default: butterfly.config)

    Example:
        # Test with 4 workers
        create_test_config(v_grid_row_workers=4, v_grid_column_workers=4)
        # Now execute_as_grid will use 4 workers

        # Test with unlimited
        create_test_config(v_grid_row_workers=0, v_grid_column_workers=0)
    """
    var content = "# Auto-generated test config\n"
    content += (
        "parallelization.v_grid_row_workers = "
        + String(v_grid_row_workers)
        + "\n"
    )
    content += (
        "parallelization.v_grid_column_workers = "
        + String(v_grid_column_workers)
        + "\n"
    )

    with open(output_path, "w") as f:
        f.write(content)
