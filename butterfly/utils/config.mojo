"""
Butterfly Configuration System

Provides file-based configuration for parallelization, performance tuning,
algorithm selection, memory management, and debugging.
"""

from sys.info import num_physical_cores


fn detect_physical_cores() -> Int:
    """Detect number of physical CPU cores, fallback to 16 if detection fails.
    """
    try:
        var cores = num_physical_cores()
        if cores > 0:
            return cores
    except:
        pass
    return 16  # Fallback


struct ButterflyConfig(Copyable):
    """Comprehensive Butterfly configuration."""

    # Parallelization
    var parallel_default: Int
    var parallel_quantum: Int
    var parallel_quantum_four_quarters: Int
    var parallel_quantum_simd_v2_chunks: Int  # Worker chunks for SIMD v2 kernels
    var parallel_fft: Int
    var parallel_fft_global: Int
    var parallel_fft_local: Int

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
        self.parallel_fft = 0
        self.parallel_fft_global = 0
        self.parallel_fft_local = 0

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
        self.parallel_fft = existing.parallel_fft
        self.parallel_fft_global = existing.parallel_fft_global
        self.parallel_fft_local = existing.parallel_fft_local

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
                if key == "parallelization.default":
                    config.parallel_default = atol(value)
                elif key == "parallelization.quantum_workers":
                    config.parallel_quantum = atol(value)
                elif key == "parallelization.quantum_four_quarters":
                    config.parallel_quantum_four_quarters = atol(value)
                elif key == "parallelization.quantum_simd_v2_chunks":
                    config.parallel_quantum_simd_v2_chunks = atol(value)
                elif key == "parallelization.fft_workers":
                    config.parallel_fft = atol(value)
                elif key == "parallelization.fft_global_stage":
                    config.parallel_fft_global = atol(value)
                elif key == "parallelization.fft_local_stage":
                    config.parallel_fft_local = atol(value)

                # Parse performance settings
                elif key == "performance.simd_width":
                    config.simd_width = atol(value)
                elif key == "performance.tile_size":
                    config.tile_size = atol(value)
                elif key == "performance.l2_tile_size":
                    config.l2_tile_size = atol(value)
                elif key == "performance.alignment":
                    config.alignment = atol(value)

                # Parse algorithm settings
                elif key == "algorithm.default_strategy":
                    config.default_strategy = String(value)
                elif key == "algorithm.fft_algorithm":
                    config.fft_algorithm = String(value)
                elif key == "algorithm.verification_enabled":
                    config.verification_enabled = value.lower() == "true"
                elif key == "algorithm.correctness_checks":
                    config.correctness_checks = value.lower() == "true"

                # Parse memory settings
                elif key == "memory.buffer_pool_size":
                    config.buffer_pool_size = atol(value)
                elif key == "memory.preallocate_buffers":
                    config.preallocate_buffers = value.lower() == "true"
                elif key == "memory.max_state_size":
                    config.max_state_size = atol(value)

                # Parse debugging settings
                elif key == "debugging.verbosity":
                    config.verbosity = atol(value)
                elif key == "debugging.timing_enabled":
                    config.timing_enabled = value.lower() == "true"
                elif key == "debugging.profile_enabled":
                    config.profile_enabled = value.lower() == "true"
                elif key == "debugging.log_file":
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
    """Load configuration from file with fallback chain."""
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


fn get_workers(operation_type: String) -> Int:
    """Get worker count for a specific operation type."""
    return get_config().get_workers(operation_type)


fn is_timing_enabled() -> Bool:
    """Check if timing is enabled."""
    return get_config().timing_enabled


fn get_verbosity() -> Int:
    """Get debug verbosity level."""
    return get_config().verbosity


fn get_default_strategy() -> String:
    """Get the default execution strategy."""
    return get_config().default_strategy
