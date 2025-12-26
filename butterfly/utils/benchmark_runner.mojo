"""
Pure Mojo benchmark runner with configurable parameters and benchmarks.

No Python dependencies - uses native Mojo collections and I/O.
"""

from time import perf_counter_ns
from collections import Dict, List
from pathlib import Path
from butterfly.utils.benchmark_utils import get_timestamp_string


fn format_float(value: Float64, decimals: Int = 2) -> String:
    """Format a float to a fixed number of decimal places."""
    var multiplier = Float64(10**decimals)
    var rounded = Int(value * multiplier + 0.5) / multiplier
    return String(rounded)


struct BenchmarkResult(ImplicitlyCopyable, Movable):
    """Single benchmark result."""

    var params: Dict[String, String]
    var benchmarks: Dict[String, Float64]

    fn __init__(out self):
        self.params = Dict[String, String]()
        self.benchmarks = Dict[String, Float64]()

    fn __copyinit__(out self, existing: Self):
        self.params = existing.params.copy()
        self.benchmarks = existing.benchmarks.copy()

    fn __moveinit__(out self, deinit existing: Self):
        self.params = existing.params^
        self.benchmarks = existing.benchmarks^


struct BenchmarkRunner:
    """Flexible benchmark runner with configurable parameters and benchmarks."""

    var suite_name: String
    var timestamp: String
    var results: List[BenchmarkResult]
    var param_columns: List[String]
    var bench_columns: List[String]

    fn __init__(out self, suite_name: String):
        self.suite_name = suite_name
        self.timestamp = get_timestamp_string()
        self.results = List[BenchmarkResult]()
        self.param_columns = List[String]()
        self.bench_columns = List[String]()

    fn log_progress(self, message: String):
        """Print a progress message with the runner's prefix."""
        print("[" + self.suite_name + "] " + message)

    fn set_param_columns(mut self, var columns: List[String]):
        """Set the parameter column names."""
        self.param_columns = columns^

    fn set_bench_columns(mut self, var columns: List[String]):
        """Set the benchmark column names."""
        self.bench_columns = columns^

    fn add_result(
        mut self,
        params: Dict[String, String],
        bench_name: String,
        time_ms: Float64,
    ) raises:
        """Add a benchmark result for a specific parameter combination."""
        # Find existing row with matching parameters
        var row_idx = -1
        for i in range(len(self.results)):
            var matches = True
            for j in range(len(self.param_columns)):
                var param_name = self.param_columns[j]
                if self.results[i].params[param_name] != params[param_name]:
                    matches = False
                    break
            if matches:
                row_idx = i
                break

        # Create new row if needed
        if row_idx == -1:
            var new_result = BenchmarkResult()
            new_result.params = params.copy()
            self.results.append(new_result^)
            row_idx = len(self.results) - 1

        # Add benchmark result
        self.results[row_idx].benchmarks[bench_name] = time_ms

    fn print_table(self, show_winner: Bool = True) raises:
        """Print results as a formatted table with dynamic column widths."""
        if len(self.results) == 0:
            print("No results to display")
            return

        # Calculate column widths
        var param_widths = Dict[String, Int]()
        var bench_widths = Dict[String, Int]()

        # Initialize with header widths
        for i in range(len(self.param_columns)):
            param_widths[self.param_columns[i]] = len(self.param_columns[i])
        for i in range(len(self.bench_columns)):
            bench_widths[self.bench_columns[i]] = len(self.bench_columns[i])

        # Calculate max widths from data
        for i in range(len(self.results)):
            var result = self.results[i]

            # Check parameter widths
            for j in range(len(self.param_columns)):
                var param = self.param_columns[j]
                var val_len = len(result.params[param])
                if val_len > param_widths[param]:
                    param_widths[param] = val_len

            # Check benchmark widths (account for "XX.XX*" format)
            for j in range(len(self.bench_columns)):
                var bench = self.bench_columns[j]
                if bench in result.benchmarks:
                    var val_len = 8  # "XXXX.XX*" max
                    if val_len > bench_widths[bench]:
                        bench_widths[bench] = val_len

        print("\n" + self.suite_name + " Results")
        print("=" * 120)

        # Build and print header
        var header = ""
        for i in range(len(self.param_columns)):
            var param = self.param_columns[i]
            var width = param_widths[param] + 2
            header += param.ljust(width) + "| "
        for i in range(len(self.bench_columns)):
            var bench = self.bench_columns[i]
            var width = bench_widths[bench] + 2
            header += bench.ljust(width) + "| "
        if show_winner:
            header += "Winner"
        print(header)
        print("-" * len(header))

        # Print rows
        for i in range(len(self.results)):
            var result = self.results[i]
            var row = ""

            # Parameter columns
            for j in range(len(self.param_columns)):
                var param = self.param_columns[j]
                var width = param_widths[param] + 2
                row += result.params[param].ljust(width) + "| "

            # Find winner
            var min_time = -1.0
            var max_time = -1.0
            var winner = ""
            for j in range(len(self.bench_columns)):
                var bench = self.bench_columns[j]
                if bench in result.benchmarks:
                    var t = result.benchmarks[bench]
                    if min_time < 0 or t < min_time:
                        min_time = t
                        winner = bench
                    if t > max_time:
                        max_time = t

            # Benchmark columns
            for j in range(len(self.bench_columns)):
                var bench = self.bench_columns[j]
                var width = bench_widths[bench] + 2
                if bench in result.benchmarks:
                    var time_val = result.benchmarks[bench]
                    var time_str = format_float(time_val)
                    if show_winner and bench == winner:
                        time_str = time_str + "*"
                    row += time_str.ljust(width) + "| "
                else:
                    row += "-".ljust(width) + "| "

            # Winner column
            if show_winner and min_time > 0:
                var winner_text = winner
                if max_time > min_time:
                    var speedup = max_time / min_time
                    var speedup_str = format_float(speedup)
                    winner_text = winner + " (" + speedup_str + "x)"
                row += winner_text

            print(row)

        print()
        if show_winner:
            print("* = fastest for this parameter combination")
            print()

    fn _get_timestamp_string(self) -> String:
        """Get timestamp string in format YYYY_MM_DD_HHMMSS."""
        return get_timestamp_string()

    fn save_csv(self, filepath: String) raises:
        """Save results to CSV with timestamp.

        Example:
            runner.save_csv("benches/results/my_benchmark")
            # Saves to: benches/results/my_benchmark_TIMESTAMP.csv
        """
        # Extract directory and filename from filepath
        var last_slash = filepath.rfind("/")
        var base_dir = filepath[:last_slash] if last_slash >= 0 else "."
        var filename_with_ext = (
            filepath[last_slash + 1 :] if last_slash >= 0 else filepath
        )
        var last_dot = filename_with_ext.rfind(".")
        var filename = (
            filename_with_ext[:last_dot] if last_dot >= 0 else filename_with_ext
        )

        # Check for results directory override from environment
        from os import getenv

        var results_dir = getenv("MOJO_BENCH_RESULTS_DIR", base_dir)

        # Create directory if it doesn't exist
        from os import makedirs

        makedirs(results_dir, exist_ok=True)

        # Build final path with timestamp
        var final_path = (
            results_dir + "/" + filename + "_" + self.timestamp + ".csv"
        )

        # Write CSV
        with open(final_path, "w") as f:
            # Write header
            var header = ""
            for i in range(len(self.param_columns)):
                header += self.param_columns[i] + ","
            for i in range(len(self.bench_columns)):
                header += self.bench_columns[i]
                if i < len(self.bench_columns) - 1:
                    header += ","
            header += ",timestamp\n"
            f.write(header)

            # Write data rows
            for i in range(len(self.results)):
                var result = self.results[i]
                var row = ""

                # Parameters
                for j in range(len(self.param_columns)):
                    var param = self.param_columns[j]
                    row += result.params[param] + ","

                # Benchmarks
                for j in range(len(self.bench_columns)):
                    var bench = self.bench_columns[j]
                    if bench in result.benchmarks:
                        row += format_float(result.benchmarks[bench])
                    # else: empty field
                    if j < len(self.bench_columns) - 1:
                        row += ","

                row += "," + String(self.timestamp) + "\n"
                f.write(row)

        print("Results saved to:", final_path)
