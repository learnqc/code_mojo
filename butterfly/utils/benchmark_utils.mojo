"""Helper functions for benchmarking and utilities."""

fn get_timestamp_string() -> String:
    """Get timestamp string in human-readable format.

    Using Unix timestamp as a zero-dependency, crash-safe alternative
    to Python's datetime.
    """
    from sys import external_call

    return String(external_call["time", Int64](0))


fn get_human_readable_timestamp() -> String:
    """Get human-readable timestamp (YYYY_MM_DD__HH_MM_SS__msecday).

    Attempts to use Python datetime for formatting.
    Falls back to Unix timestamp if Python unavailable or fails.

    Returns:
        Timestamp string like "2025_12_29__17_32_00__12345678" or Unix timestamp on failure.
    """
    try:
        from python import Python

        var datetime = Python.import_module("datetime")
        var now = datetime.datetime.now()
        var date_str = String(now.strftime("%Y_%m_%d"))
        var time_str = String(now.strftime("%H_%M_%S"))
        var msecday = (
            (now.hour * 3600 + now.minute * 60 + now.second) * 1000
            + now.microsecond // 1000
        )
        var msecday_str = String(msecday).rjust(8, "0")
        return date_str + "__" + time_str + "__" + msecday_str
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
        ```var (id, name) = parse_benchmark_args(
            "my_benchmark",
            "My Benchmark"
        )
        ```
    """
    # TODO: Parse sys.argv when Mojo supports it
    # Look for --benchmark-id and --display-name flags
    # For now, return defaults
    return (default_id, default_name)
