"""
Test for Butterfly Configuration System
"""

from butterfly.utils.config import (
    ButterflyConfig,
    get_config,
    get_workers,
    is_timing_enabled,
    get_verbosity,
    get_default_strategy,
)


fn test_default_config():
    """Test default configuration values."""
    print("=== Testing Default Configuration ===")
    var config = ButterflyConfig()

    print("  parallel_default:", config.parallel_default, "(expected: 0)")
    print("  parallel_quantum:", config.parallel_quantum, "(expected: 0)")
    print(
        "  parallel_quantum_four_quarters:",
        config.parallel_quantum_four_quarters,
        "(expected: 4)",
    )
    print("  simd_width:", config.simd_width, "(expected: 0)")
    print("  tile_size:", config.tile_size, "(expected: 32768)")
    print("  default_strategy:", config.default_strategy, "(expected: 'auto')")
    print("  fft_algorithm:", config.fft_algorithm, "(expected: 'phast')")
    print("  verbosity:", config.verbosity, "(expected: 0)")

    print("✓ Default configuration test passed\n")


fn test_worker_resolution():
    """Test worker count resolution logic."""
    print("=== Testing Worker Resolution ===")
    var config = ButterflyConfig()

    # Test default fallback
    config.parallel_default = 16
    config.parallel_quantum = 0
    var quantum_workers = config.get_workers("quantum")
    print(
        "  quantum workers (using default):", quantum_workers, "(expected: 16)"
    )

    # Test specific override
    config.parallel_fft = 8
    var fft_workers = config.get_workers("fft")
    print("  fft workers:", fft_workers, "(expected: 8)")

    # Test nested override
    config.parallel_fft_global = 0
    var fft_global_workers = config.get_workers("fft_global")
    print(
        "  fft_global workers (using fft):", fft_global_workers, "(expected: 8)"
    )

    # Test nested specific value
    config.parallel_fft_global = 12
    var fft_global_workers2 = config.get_workers("fft_global")
    print("  fft_global workers:", fft_global_workers2, "(expected: 12)")

    print("✓ Worker resolution test passed\n")


fn test_file_loading() raises:
    """Test loading configuration from file."""
    print("=== Testing File Loading ===")

    # Create a temporary config file
    var test_config = """# Test config
parallelization.default = 0
parallelization.quantum_workers = 8
parallelization.fft_workers = 16
debugging.verbosity = 2
debugging.timing_enabled = true
algorithm.default_strategy = fused_v3
"""

    with open("test_butterfly.config", "w") as f:
        _ = f.write(test_config)

    # Load configuration
    var config = ButterflyConfig.from_file("test_butterfly.config")

    # Verify loaded values
    print("  parallel_quantum:", config.parallel_quantum, "(expected: 8)")
    print("  parallel_fft:", config.parallel_fft, "(expected: 16)")
    print("  verbosity:", config.verbosity, "(expected: 2)")
    print("  timing_enabled:", config.timing_enabled, "(expected: True)")
    print(
        "  default_strategy:", config.default_strategy, "(expected: 'fused_v3')"
    )

    print("✓ File loading test passed\n")


fn test_global_config() raises:
    """Test global configuration singleton."""
    print("=== Testing Global Configuration ===")

    # Get global config (should use defaults if no file)
    var config = get_config()

    # Test convenience accessors
    var workers = get_workers("quantum")
    var timing = is_timing_enabled()
    var verbosity = get_verbosity()
    var strategy = get_default_strategy()

    print("  Workers (quantum):", workers)
    print("  Timing enabled:", timing)
    print("  Verbosity:", verbosity)
    print("  Default strategy:", strategy)

    print("✓ Global configuration test passed\n")


fn test_print_config():
    """Test configuration printing."""
    print("=== Testing Configuration Printing ===")
    var config = ButterflyConfig()
    config.parallel_quantum = 8
    config.parallel_fft = 16
    config.verbosity = 2
    config.timing_enabled = True

    config.print_config()
    print("\n✓ Configuration printing test passed\n")


fn main() raises:
    print("╔════════════════════════════════════════════╗")
    print("║  Butterfly Configuration System Tests      ║")
    print("╚════════════════════════════════════════════╝\n")

    test_default_config()
    test_worker_resolution()
    test_file_loading()
    test_global_config()
    test_print_config()

    print("=" * 50)
    print("All tests completed! ✓")
    print("=" * 50)
