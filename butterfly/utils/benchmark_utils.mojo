# Helper functions for timestamp generation


fn get_timestamp_string() -> String:
    """Get timestamp string (nanoseconds since program start).

    Note: Mojo doesn't have access to system time yet, so this is just
    a monotonic counter for uniqueness. Use Python report generator for
    human-readable dates.
    """
    from time import perf_counter_ns

    return String(Int(perf_counter_ns()))
