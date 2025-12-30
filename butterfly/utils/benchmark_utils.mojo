"""Helper functions for benchmarking and utilities."""

from butterfly.core.state import QuantumState
from butterfly.core.circuit import QuantumCircuit
from butterfly.core.execution_strategy import (
    ExecutionStrategy,
    get_strategy_name,
    get_strategy_description,
)
from collections import List


fn get_timestamp_string() -> String:
    """Get timestamp string in human-readable format.

    Using Unix timestamp as a zero-dependency, crash-safe alternative
    to Python's datetime.
    """
    from sys import external_call

    return String(external_call["time", Int64](0))


fn get_human_readable_timestamp() -> String:
    """Get human-readable timestamp (YYYY_MM_DD_HHMMSS).

    Attempts to use Python datetime for formatting.
    Falls back to Unix timestamp if Python unavailable or fails.

    Returns:
        Timestamp string like "2025_12_29_173200" or Unix timestamp on failure.
    """
    try:
        from python import Python

        var datetime = Python.import_module("datetime")
        var now = datetime.datetime.now()
        var formatted = now.strftime("%Y_%m_%d_%H%M%S")
        return String(formatted)
    except:
        # Fallback to Unix timestamp if Python fails
        return get_timestamp_string()


fn get_date_string() -> String:
    """Get current date (Unix timestamp) as a zero-dependency alternative."""
    from sys import external_call

    return String(external_call["time", Int64](0))


fn parse_benchmark_args(
    default_id: String,
    default_name: String,
) -> Tuple[String, String]:
    """Parse benchmark arguments from command line.

    When Mojo supports sys.argv, this will parse --benchmark-id and --display-name.
    For now, returns the defaults.

    Args:
        default_id: Default benchmark ID (e.g., "value_encoding_strategies").
        default_name: Default display name (e.g., "Value Encoding Strategies").

    Returns:
        Tuple of (benchmark_id, display_name).

    Example:
        var (id, name) = parse_benchmark_args(
            "my_benchmark",
            "My Benchmark"
        )
    """
    # TODO: Parse sys.argv when Mojo supports it
    # Look for --benchmark-id and --display-name flags
    # For now, return defaults
    return (default_id, default_name)


struct BenchmarkableFunction:
    """Wraps a callable with metadata for benchmarking.

    This allows any function that produces a QuantumState to be:
    - Named (for reporting)
    - Described (optional, defaults to name)
    - Verified (compared against reference)
    - Benchmarked (timed)
    """

    var name: String
    var description: String
    var _execute: fn (QuantumCircuit) -> QuantumState

    fn __init__(
        inoutself,
        name: String,
        execute_fn: fn (QuantumCircuit) -> QuantumState,
        description: String = "",
    ):
        """Create a benchmarkable function.

        Args:
            name: Function name (used in reports).
            execute_fn: Function that takes a circuit and returns a state.
            description: Optional description (defaults to name).
        """
        self.name = name
        self.description = description if description else name
        self._execute = execute_fn

    fn execute(self, circuit: QuantumCircuit) -> QuantumState:
        """Execute the wrapped function."""
        return self._execute(circuit)


fn wrap_strategy(strategy: ExecutionStrategy) -> BenchmarkableFunction:
    """Wrap an execution strategy as a benchmarkable function.

    Args:
        strategy: The execution strategy to wrap.

    Returns:
        BenchmarkableFunction with strategy name and description.
    """
    var name = get_strategy_name(strategy)
    var description = get_strategy_description(strategy)

    # Create closure that captures the strategy
    fn execute_with_strategy(circuit: QuantumCircuit) -> QuantumState:
        var c = circuit.copy()
        return c.run_with_strategy(strategy)

    return BenchmarkableFunction(name, execute_with_strategy, description)


fn wrap_strategies(
    strategies: List[ExecutionStrategy],
) -> List[BenchmarkableFunction]:
    """Convert a list of strategies to benchmarkable functions.

    Args:
        strategies: List of execution strategies.

    Returns:
        List of benchmarkable functions.
    """
    var functions = List[BenchmarkableFunction]()
    for i in range(len(strategies)):
        functions.append(wrap_strategy(strategies[i]))
    return functions^
