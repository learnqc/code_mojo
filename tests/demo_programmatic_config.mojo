"""
Demo showing programmatic config loading for tests.

This approach works WITHIN a running program, unlike environment variables
which must be set before the process starts.
"""

from butterfly.utils.config import (
    load_config_from_path,
    get_workers_from_config,
)


fn test_with_config(config_path: String) raises:
    """Run a test with a specific config."""
    print("\n" + "=" * 60)
    print("Testing with config:", config_path)
    print("=" * 60)

    # Load config programmatically
    var config = load_config_from_path(config_path)

    # Query config
    var row_workers = get_workers_from_config(config, "v_grid_rows")
    var col_workers = get_workers_from_config(config, "v_grid_columns")

    print("  v_grid_row_workers:", row_workers)
    print("  v_grid_column_workers:", col_workers)

    # Simulate running a test with this config
    if row_workers == 0:
        print("  ✓ Test mode: UNLIMITED parallelization")
    else:
        print("  ✓ Test mode: LIMITED to", row_workers, "workers")


fn main() raises:
    print("Programmatic Config Loading Demo")
    print("This works for tests that need to switch configs dynamically!\n")

    # Test with different configs in sequence
    test_with_config("configs/workers_2.config")
    test_with_config("configs/workers_8.config")
    test_with_config("configs/unlimited.config")

    print("\n" + "=" * 60)
    print("✓ All config variations tested in one run!")
    print("=" * 60)

    print("\nUsage in tests:")
    print("  var config = load_config_from_path('configs/workers_2.config')")
    print("  var workers = get_workers_from_config(config, 'v_grid_rows')")
