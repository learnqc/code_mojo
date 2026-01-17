"""
Thread monitoring utilities for Mojo benchmarks.

Provides cross-platform thread count monitoring for performance analysis.
Currently supports macOS via ps command.
"""

from time import perf_counter_ns, sleep
from butterfly.core.types import FloatType


fn get_current_pid() -> Int:
    """Get the process ID of the current Mojo process.

    Returns:
        Process ID, or -1 on failure.
    """
    # Use libc getpid()
    from sys.ffi import external_call

    var pid = external_call["getpid", Int]()
    return pid


fn get_thread_count(pid: Int) -> Int:
    """Get the number of threads for a given process ID.

    Args:
        pid: Process ID to query.

    Returns:
        Number of threads, or -1 on failure.

    Note:
        Currently only supports macOS using 'ps -M <pid>' command.
    """
    if pid <= 0:
        return -1

    try:
        # On macOS, use ps -M to list threads
        # Count lines minus 1 for header
        from python import Python

        var subprocess = Python.import_module("subprocess")

        var cmd = String("ps -M ") + String(pid) + String(" | wc -l")
        var result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True
        )

        var output = String(result.stdout).strip()
        var count = atol(output)

        # ps output includes header line, so subtract 1
        # Also includes the process itself as one line
        return Int(count) - 1 if count > 1 else 1
    except:
        return -1


fn get_current_thread_count() -> Int:
    """Get thread count for the current process.

    Returns:
        Number of threads, or -1 on failure.
    """
    var pid = get_current_pid()
    return get_thread_count(pid)


struct ThreadSnapshot(ImplicitlyCopyable, Movable):
    """Snapshot of thread count at a specific time."""

    var timestamp_ns: Int
    var thread_count: Int

    fn __init__(out self, timestamp_ns: Int, thread_count: Int):
        self.timestamp_ns = timestamp_ns
        self.thread_count = thread_count

    fn __copyinit__(out self, existing: Self):
        self.timestamp_ns = existing.timestamp_ns
        self.thread_count = existing.thread_count

    fn __moveinit__(out self, deinit existing: Self):
        self.timestamp_ns = existing.timestamp_ns
        self.thread_count = existing.thread_count


struct ThreadStats(ImplicitlyCopyable, Movable):
    """Statistics from sampling thread counts."""

    var min_threads: Int
    var max_threads: Int
    var avg_threads: FloatType
    var sample_count: Int

    fn __init__(out self):
        self.min_threads = -1
        self.max_threads = -1
        self.avg_threads = 0.0
        self.sample_count = 0

    fn __init__(
        out self,
        min_threads: Int,
        max_threads: Int,
        avg_threads: FloatType,
        sample_count: Int,
    ):
        self.min_threads = min_threads
        self.max_threads = max_threads
        self.avg_threads = avg_threads
        self.sample_count = sample_count

    fn __copyinit__(out self, existing: Self):
        self.min_threads = existing.min_threads
        self.max_threads = existing.max_threads
        self.avg_threads = existing.avg_threads
        self.sample_count = existing.sample_count

    fn __moveinit__(out self, deinit existing: Self):
        self.min_threads = existing.min_threads
        self.max_threads = existing.max_threads
        self.avg_threads = existing.avg_threads
        self.sample_count = existing.sample_count


fn compute_thread_stats(snapshots: List[ThreadSnapshot]) -> ThreadStats:
    """Compute statistics from a list of thread snapshots.

    Args:
        snapshots: List of thread count snapshots.

    Returns:
        Statistics including min, max, and average thread counts.
    """
    if len(snapshots) == 0:
        return ThreadStats()

    var min_threads = snapshots[0].thread_count
    var max_threads = snapshots[0].thread_count
    var total = 0

    for i in range(len(snapshots)):
        var count = snapshots[i].thread_count
        if count > 0:  # Skip error values
            if count < min_threads or min_threads < 0:
                min_threads = count
            if count > max_threads:
                max_threads = count
            total += count

    var avg_threads = FloatType(total) / FloatType(len(snapshots))

    return ThreadStats(min_threads, max_threads, avg_threads, len(snapshots))


fn sample_current_threads(
    num_samples: Int = 3, interval_ms: Int = 10
) raises -> ThreadStats:
    """Sample thread count for the current process multiple times.

    Args:
        num_samples: Number of samples to take.
        interval_ms: Milliseconds between samples.

    Returns:
        Thread statistics from sampling.
    """
    var snapshots = List[ThreadSnapshot]()

    for i in range(num_samples):
        var timestamp = Int(perf_counter_ns())
        var count = get_current_thread_count()
        snapshots.append(ThreadSnapshot(timestamp, count))

        if i < num_samples - 1:
            sleep(interval_ms / 1000.0)

    return compute_thread_stats(snapshots)


fn format_thread_stats(stats: ThreadStats) -> String:
    """Format thread statistics as a human-readable string.

    Args:
        stats: Thread statistics to format.

    Returns:
        Formatted string like "min:4 max:8 avg:6.2 (n=5)".
    """
    if stats.sample_count == 0:
        return "no data"

    return (
        "min:"
        + String(stats.min_threads)
        + " max:"
        + String(stats.max_threads)
        + " avg:"
        + String(round(stats.avg_threads, 1))
        + " (n="
        + String(stats.sample_count)
        + ")"
    )
