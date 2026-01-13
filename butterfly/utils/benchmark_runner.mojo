"""
Pure Mojo benchmark runner with configurable parameters and benchmarks.

No Python dependencies - uses native Mojo collections and I/O.
"""

from time import perf_counter_ns
from collections import Dict, List
from pathlib import Path
from butterfly.utils.benchmark_utils import get_timestamp_string
from benchmark import run, Unit


fn format_float(value: Float64, decimals: Int = 2) -> String:
    """Format a float to a fixed number of decimal places."""
    # var multiplier = Float64(10**decimals)
    # var rounded = Int(value * multiplier + 0.5) / multiplier
    rounded_str = String(round(value, decimals))
    return rounded_str


fn median(values: List[Float64]) -> Float64:
    if len(values) == 0:
        return 0.0
    var sorted = values.copy()
    # Insertion sort is fine for small trial counts.
    for i in range(1, len(sorted)):
        var key = sorted[i]
        var j = i - 1
        while j >= 0 and sorted[j] > key:
            sorted[j + 1] = sorted[j]
            j -= 1
        sorted[j + 1] = key
    var mid = len(sorted) // 2
    if len(sorted) % 2 == 1:
        return sorted[mid]
    return (sorted[mid - 1] + sorted[mid]) / 2.0


fn list_recent_csvs(
    results_dir: String, prefix: String, max_files: Int
) -> List[String]:
    var files = List[String]()
    try:
        from python import Python

        var os = Python.import_module("os")
        var entries = os.listdir(results_dir)
        for i in range(len(entries)):
            var name = String(entries[i])
            var full = results_dir + "/" + name
            try:
                if os.path.isdir(full):
                    var sub_entries = os.listdir(full)
                    for j in range(len(sub_entries)):
                        var sub_name = String(sub_entries[j])
                        if not sub_name.endswith(".csv"):
                            continue
                        if not sub_name.startswith(prefix + "_"):
                            continue
                        files.append(full + "/" + sub_name)
                else:
                    if not name.endswith(".csv"):
                        continue
                    if not name.startswith(prefix + "_"):
                        continue
                    files.append(full)
            except:
                continue
    except:
        return files^

    # Sort newest-first (lex order matches timestamp format).
    for i in range(1, len(files)):
        var key = files[i]
        var j = i - 1
        while j >= 0 and files[j] < key:
            files[j + 1] = files[j]
            j -= 1
        files[j + 1] = key

    if len(files) > max_files:
        var trimmed = List[String](capacity=max_files)
        for i in range(max_files):
            trimmed.append(files[i])
        return trimmed^
    return files^


fn perf_function_call_ms[
    Input: AnyType & Copyable & Movable, Return: AnyType
](
    f: fn (Input) raises -> Return,
    input: Input,
    iters: Int = 3,
    decimals: Int = 3,
    warmup_iters: Int = 0,
    trials: Int = 1,
) raises -> Float64:
    for _ in range(warmup_iters):
        _ = f(input)

    var times = List[Float64](capacity=trials)
    for _ in range(trials):
        var t0 = Int(perf_counter_ns())
        for _ in range(iters):
            _ = f(input)
        var t1 = Int(perf_counter_ns())
        times.append(Float64(t1 - t0) / 1_000_000.0 / iters)

    return round(median(times), decimals)


fn should_autosave() -> Bool:
    """Check if --autosave flag is present in command-line arguments.

    Returns True if --autosave is present, False otherwise.
    Default is False (don't save during development).
    """
    from sys import argv

    for i in range(len(argv())):
        if argv()[i] == "--autosave":
            return True
    return False


fn bench_function_call_ms[
    Input: AnyType, Return: AnyType
](
    f: fn (Input) raises -> Return,
    input: Input,
    iters: Int = 5,
    decimals: Int = 3,
) raises -> Float64:
    @parameter
    fn bench():
        try:
            _ = f(input)
        except e:
            pass

    var t = run[bench](2, iters).mean(Unit.ms)
    return round(t, decimals)


fn create_runner(
    name: String,
    description: String,
    mut param_cols: List[String],
    mut bench_cols: List[String],
    reference_idx: Int = 0,
) raises -> BenchmarkRunner:
    var runner = BenchmarkRunner(description)
    runner.set_param_columns(param_cols.copy())
    runner.set_bench_columns(bench_cols.copy())
    runner.set_reference_idx(reference_idx)
    print(description)
    print("=" * 80)
    return runner^


@fieldwise_init
struct LabeledFunction[Input: AnyType, Return: AnyType](
    Copyable, ImplicitlyCopyable, Movable
):
    """Encapsulates a function and a name for automated verification and benchmarking.
    """

    var name: String
    var func: fn (Input) raises -> Return

    fn __copyinit__(out self, existing: Self):
        self.name = existing.name
        self.func = existing.func

    fn __moveinit__(out self, deinit existing: Self):
        self.name = existing.name^
        self.func = existing.func

    fn run(self, input: Input) raises -> Return:
        """Execute the wrapped function."""
        return self.func(input)


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


struct BenchmarkRunner(Movable):
    """Flexible benchmark runner with configurable parameters and benchmarks."""

    var suite_name: String
    var timestamp: String
    var results: List[BenchmarkResult]
    var param_columns: List[String]
    var bench_columns: List[String]
    var reference_idx: Int

    fn __init__(out self, suite_name: String) raises:
        self.suite_name = suite_name

        # Use human-readable timestamp with fallback to Unix timestamp
        from butterfly.utils.benchmark_utils import get_human_readable_timestamp

        self.timestamp = get_human_readable_timestamp()

        self.results = List[BenchmarkResult]()
        self.param_columns = List[String]()
        self.bench_columns = List[String]()
        self.reference_idx = 0

    fn log_progress(self, message: String):
        """Print a progress message with the runner's prefix."""
        # print("[" + self.suite_name + "] " + message)
        print(message)

    fn set_param_columns(mut self, var columns: List[String]):
        """Set the parameter column names."""
        self.param_columns = columns^

    fn set_bench_columns(mut self, var columns: List[String]):
        """Set the benchmark column names."""
        self.bench_columns = columns^

    fn set_reference_idx(mut self, idx: Int):
        """Set the reference column index."""
        self.reference_idx = idx

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
                if (
                    param_name not in params
                    or param_name not in self.results[i].params
                ):
                    matches = False
                    break
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
        var param_str = ""
        for k in params.keys():
            param_str += String(k) + "=" + String(params[k]) + ", "
        print(param_str, bench_name.ljust(15), "->", round(time_ms, 2))

    fn add_perf_result[
        Input: AnyType & Copyable & Movable, Return: AnyType
    ](
        mut self,
        params: Dict[String, String],
        name: String,
        func: fn (Input) raises -> Return,
        input: Input,
        iters: Int = 5,
        decimals: Int = 3,
        warmup_iters: Int = 0,
        trials: Int = 1,
    ) raises:
        var t = perf_function_call_ms(
            func, input, iters, decimals, warmup_iters, trials
        )
        self.add_result(params, name, t)

    fn add_bench_result[
        Input: AnyType & Copyable & Movable, Return: AnyType
    ](
        mut self,
        params: Dict[String, String],
        name: String,
        func: fn (Input) raises -> Return,
        input: Input,
        iters: Int = 5,
        decimals: Int = 3,
    ) raises:
        var t = bench_function_call_ms(func, input, iters, decimals)
        self.add_result(params, name, t)

    fn add_perf_result_with_threads[
        Input: AnyType & Copyable & Movable, Return: AnyType
    ](
        mut self,
        params: Dict[String, String],
        name: String,
        func: fn (Input) raises -> Return,
        input: Input,
        iters: Int = 5,
        decimals: Int = 3,
        warmup_iters: Int = 0,
        trials: Int = 1,
        thread_samples: Int = 3,
    ) raises:
        """Add performance result with thread count monitoring.

        Args:
            params: Parameter dictionary.
            name: Benchmark name.
            func: Function to benchmark.
            input: Input to the function.
            iters: Number of iterations.
            decimals: Decimal places for timing.
            thread_samples: Number of thread samples to take during execution.
        """
        from butterfly.utils.thread_monitor import (
            sample_current_threads,
            format_thread_stats,
        )

        # Sample threads before
        var stats_before = sample_current_threads(
            num_samples=thread_samples, interval_ms=10
        )

        # Run the benchmark
        var t = perf_function_call_ms(
            func, input, iters, decimals, warmup_iters, trials
        )

        # Sample threads after
        var stats_after = sample_current_threads(
            num_samples=thread_samples, interval_ms=10
        )

        # Log the result with thread info
        self.add_result(params, name, t)
        var thread_info = (
            "  [threads: " + format_thread_stats(stats_after) + "]"
        )
        print("  " + thread_info)

    # --- AUTOMATED BATCH PROCESSING ---

    fn verify[
        Return: AnyType & Copyable & Movable & ImplicitlyCopyable
    ](
        mut self,
        values: List[Return],
        names: List[String],
        comparator: fn (Return, Return, Float64) raises,
        baseline_idx: Int = 0,
        stop_on_failure: Bool = True,
        tolerance: Float64 = 1e-5,
    ) raises:
        """Verify a list of pre-computed values against a baseline."""
        if len(values) == 0:
            return

        var base_val = values[baseline_idx]
        var base_name = names[baseline_idx]

        for i in range(len(values)):
            if i == baseline_idx:
                continue

            var contender_val = values[i]
            var contender_name = names[i]

            try:
                comparator(base_val, contender_val, tolerance)
                self.log_progress(
                    "✓ Verification of "
                    + contender_name
                    + " vs "
                    + base_name
                    + " successful"
                )
            except e:
                self.log_progress(
                    "!! Verification of "
                    + contender_name
                    + " vs "
                    + base_name
                    + " FAILED: "
                    + String(e)
                )
                if stop_on_failure:
                    raise e

    fn verify[
        Input: AnyType & Copyable & Movable,
        Return: AnyType & Copyable & Movable & ImplicitlyCopyable,
    ](
        mut self,
        input: Input,
        functions: List[LabeledFunction[Input, Return]],
        comparator: fn (Return, Return, Float64) raises,
        params: Dict[String, String] = Dict[String, String](),
        baseline_idx: Int = 0,
        stop_on_failure: Bool = True,
        tolerance: Float64 = 1e-5,
    ) raises:
        """Execute and verify a list of functions against a baseline.
        If params is provided, also records performance results.
        This implementation is memory-efficient and only keeps the baseline
        and the current contender in memory.
        """
        # 1. Verification
        var baseline_val = functions[baseline_idx].run(input)
        var baseline_name = functions[baseline_idx].name

        for i in range(len(functions)):
            var contender_name = functions[i].name
            if i == baseline_idx:
                continue

            try:
                self.log_progress("  Verifying " + contender_name + "...")
                var contender_val = functions[i].run(input)
                comparator(baseline_val, contender_val, tolerance)
                self.log_progress(
                    "✓ Verification of "
                    + contender_name
                    + " vs "
                    + baseline_name
                    + " successful"
                )
            except e:
                self.log_progress(
                    "!! Verification of "
                    + contender_name
                    + " vs "
                    + baseline_name
                    + " FAILED: "
                    + String(e)
                )
                if stop_on_failure:
                    raise e

    fn add_perf_results[
        Input: AnyType & Copyable & Movable, Return: AnyType
    ](
        mut self,
        params: Dict[String, String],
        functions: List[LabeledFunction[Input, Return]],
        input: Input,
        iters: Int = 5,
        decimals: Int = 3,
        warmup_iters: Int = 0,
        trials: Int = 1,
    ) raises:
        """Measure and record performance for a list of labeled functions."""
        for i in range(len(functions)):
            var f = functions[i]
            self.add_perf_result(
                params,
                f.name,
                f.func,
                input,
                iters,
                decimals,
                warmup_iters,
                trials,
            )

    fn add_bench_results[
        Input: AnyType & Copyable & Movable, Return: AnyType
    ](
        mut self,
        params: Dict[String, String],
        functions: List[LabeledFunction[Input, Return]],
        input: Input,
        iters: Int = 5,
        decimals: Int = 3,
    ) raises:
        """Measure and record high-precision performance for a list of labeled functions.
        """
        for i in range(len(functions)):
            var f = functions[i]
            self.add_bench_result(
                params, f.name, f.func, input, iters, decimals
            )

    # --- AGNOSTIC VERIFICATION (Hook Pattern) ---

    fn verify[
        Input: AnyType & Copyable & Movable, Return: AnyType
    ](
        mut self,
        input: Input,
        f1: fn (Input) raises -> Return,
        f2: fn (Input) raises -> Return,
        compare: fn (Return, Return, Float64) raises,
        name1: String = "func1",
        name2: String = "func2",
        tolerance: Float64 = 1e-5,
    ) raises:
        """Verify two functions using a user-provided comparison hook."""
        # self.log_progress("Verifying " + name1 + " vs " + name2 + "...")
        var val1 = f1(input)
        var val2 = f2(input)
        compare(val1, val2, tolerance)
        self.log_progress(
            "✓ Verification of " + name1 + " vs " + name2 + " successful"
        )

    # --- Built-in Primitives Overloads ---

    fn verify[
        Input: AnyType & Copyable & Movable
    ](
        mut self,
        input: Input,
        f1: fn (Input) raises -> Int,
        f2: fn (Input) raises -> Int,
        name1: String = "func1",
        name2: String = "func2",
        tolerance: Float64 = 1e-5,
    ) raises:
        """Agnostic verification for Int."""
        # self.log_progress("Verifying " + name1 + " vs " + name2 + " ...")
        var v1 = f1(input)
        var v2 = f2(input)
        if v1 != v2:
            raise Error(
                "Verification failed: " + String(v1) + " != " + String(v2)
            )
        self.log_progress(
            "✓ Verification of " + name1 + " vs " + name2 + " successful"
        )

    fn verify[
        Input: AnyType & Copyable & Movable
    ](
        mut self,
        input: Input,
        f1: fn (Input) raises -> Float64,
        f2: fn (Input) raises -> Float64,
        name1: String = "func1",
        name2: String = "func2",
        tolerance: Float64 = 1e-5,
    ) raises:
        """Agnostic verification for Float64."""
        # self.log_progress("Verifying " + name1 + " vs " + name2 + "...")
        var v1 = f1(input)
        var v2 = f2(input)
        if abs(v1 - v2) > tolerance:
            raise Error(
                "Verification failed: " + String(v1) + " != " + String(v2)
            )
        self.log_progress(
            "✓ Verification of " + name1 + " vs " + name2 + " successful"
        )

    fn verify[
        Input: AnyType & Copyable & Movable
    ](
        mut self,
        input: Input,
        f1: fn (Input) raises -> String,
        f2: fn (Input) raises -> String,
        name1: String = "func1",
        name2: String = "func2",
        tolerance: Float64 = 1e-5,
    ) raises:
        """Agnostic verification for String."""
        # self.log_progress("Verifying " + name1 + " vs " + name2 + "...")
        var v1 = f1(input)
        var v2 = f2(input)
        if v1 != v2:
            raise Error("Verification failed: '" + v1 + "' != '" + v2 + "'")
        self.log_progress(
            "✓ Verification of " + name1 + " vs " + name2 + " successful"
        )

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
            var bench_ref_name = self.bench_columns[self.reference_idx]
            header += "Winner (over " + bench_ref_name + ")"

        print("=" * len(header))
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
                var ref_name = self.bench_columns[self.reference_idx]
                var winner_text = winner
                if ref_name in result.benchmarks:
                    var bench_ref_time = result.benchmarks[ref_name]
                    if max_time > min_time:
                        var speedup = bench_ref_time / min_time
                        var speedup_str = format_float(speedup)
                        winner_text = (
                            winner.ljust(12) + " x " + speedup_str.rjust(6) + ""
                        )
                row += winner_text

            print(row)

        print()
        if show_winner:
            print("* = fastest for this parameter combination")
            print()

    fn print_history_speedup_table(
        self,
        results_dir: String,
        prefix: String,
        recent_count: Int = 5,
    ) raises:
        """Print speedup vs average of most recent CSVs.

        Speedup is computed as (historical_avg / current_time).
        """
        if len(self.results) == 0:
            print("No results to compare")
            return

        from os import getenv

        var results_root = results_dir
        if results_root == "":
            results_root = getenv("MOJO_BENCH_RESULTS_DIR", "benches/results")

        var files = list_recent_csvs(results_root, prefix, recent_count)
        if len(files) == 0:
            print("No historical results found in:", results_root)
            return

        var sums = Dict[String, Float64]()
        var counts = Dict[String, Int]()

        for fpath in files:
            try:
                with open(fpath, "r") as f:
                    var content = f.read()
                    var lines = content.split("\n")
                    if len(lines) < 2:
                        continue

                    var header = lines[0].split(",")
                    var header_idx = Dict[String, Int]()
                    for i in range(len(header)):
                        var name = String(header[i].strip())
                        if name == "" or name == "timestamp":
                            continue
                        header_idx[name] = i

                    # Ensure all param columns exist.
                    var missing_param = False
                    for i in range(len(self.param_columns)):
                        if self.param_columns[i] not in header_idx:
                            missing_param = True
                            break
                    if missing_param:
                        continue

                    for i in range(1, len(lines)):
                        var line = lines[i].strip()
                        if line == "":
                            continue

                        var row = line.split(",")
                        if len(row) < len(header):
                            continue

                        var param_key = ""
                        for j in range(len(self.param_columns)):
                            if j > 0:
                                param_key += "|"
                            var col = self.param_columns[j]
                            var idx = header_idx[col]
                            param_key += String(row[idx].strip())

                        for j in range(len(self.bench_columns)):
                            var bench = self.bench_columns[j]
                            if bench not in header_idx:
                                continue
                            var idx = header_idx[bench]
                            var val_str = row[idx].strip()
                            if val_str == "":
                                continue
                            try:
                                var val = Float64(String(val_str))
                                var key = param_key + "|" + bench
                                if key in sums:
                                    sums[key] = sums[key] + val
                                    counts[key] = counts[key] + 1
                                else:
                                    sums[key] = val
                                    counts[key] = 1
                            except:
                                pass
            except:
                continue

        if len(sums) == 0:
            print("No comparable historical cells found in:", results_root)
            return

        # Calculate column widths
        var param_widths = Dict[String, Int]()
        var bench_widths = Dict[String, Int]()

        for i in range(len(self.param_columns)):
            param_widths[self.param_columns[i]] = len(self.param_columns[i])
        for i in range(len(self.bench_columns)):
            bench_widths[self.bench_columns[i]] = len(self.bench_columns[i])

        for i in range(len(self.results)):
            var result = self.results[i]
            for j in range(len(self.param_columns)):
                var param = self.param_columns[j]
                var val_len = len(result.params[param])
                if val_len > param_widths[param]:
                    param_widths[param] = val_len

            for j in range(len(self.bench_columns)):
                var bench = self.bench_columns[j]
                if bench in result.benchmarks:
                    var time_val = result.benchmarks[bench]
                    var key = ""
                    for k in range(len(self.param_columns)):
                        if k > 0:
                            key += "|"
                        key += result.params[self.param_columns[k]].copy()
                    key = key + "|" + bench
                    if key in sums and time_val > 0:
                        var avg = sums[key] / Float64(counts[key])
                        var speedup = avg / time_val
                        var val_len = len(format_float(speedup))
                        if val_len > bench_widths[bench]:
                            bench_widths[bench] = val_len

        print("\n" + self.suite_name + " Speedup vs Recent Avg")

        var header = ""
        for i in range(len(self.param_columns)):
            var param = self.param_columns[i]
            var width = param_widths[param] + 2
            header += param.ljust(width) + "| "
        for i in range(len(self.bench_columns)):
            var bench = self.bench_columns[i]
            var width = bench_widths[bench] + 2
            header += bench.ljust(width) + "| "
        print("=" * len(header))
        print(header)
        print("-" * len(header))

        for i in range(len(self.results)):
            var result = self.results[i]
            var row = ""
            for j in range(len(self.param_columns)):
                var param = self.param_columns[j]
                var width = param_widths[param] + 2
                row += result.params[param].ljust(width) + "| "

            for j in range(len(self.bench_columns)):
                var bench = self.bench_columns[j]
                var width = bench_widths[bench] + 2
                if bench in result.benchmarks:
                    var time_val = result.benchmarks[bench]
                    var key = ""
                    for k in range(len(self.param_columns)):
                        if k > 0:
                            key += "|"
                        key += result.params[self.param_columns[k]].copy()
                    key = key + "|" + bench
                    if key in sums and time_val > 0:
                        var avg = sums[key] / Float64(counts[key])
                        var speedup = avg / time_val
                        var speedup_str = format_float(speedup)
                        row += speedup_str.ljust(width) + "| "
                    else:
                        row += "-".ljust(width) + "| "
                else:
                    row += "-".ljust(width) + "| "

            print(row)

        print()
        print(
            "speedup = historical_avg / current_time (higher is faster)"
        )

    fn _get_timestamp_string(self) raises -> String:
        """Get timestamp string in format YYYY_MM_DD_HHMMSS."""
        return get_timestamp_string()

    fn load_csv(mut self, filepath: String) raises:
        """Load results from a CSV file and merge with existing results.

        The CSV must have columns matching the runner's param_columns.
        Any other columns (except 'timestamp') are treated as benchmark results.
        """
        with open(filepath, "r") as f:
            var content = f.read()
            var lines = content.split("\n")
            if len(lines) < 2:
                return

            var header = lines[0].split(",")

            # Map column names to indices
            var param_indices = List[Int]()
            var param_names = List[String]()
            var bench_indices = List[Int]()
            var bench_names = List[String]()

            for i in range(len(header)):
                var col_name = String(header[i].strip())
                if col_name == "timestamp" or col_name == "":
                    continue

                # Check if it's a known parameter column
                var is_param = False
                for j in range(len(self.param_columns)):
                    if self.param_columns[j] == col_name:
                        is_param = True
                        break

                if is_param:
                    param_indices.append(i)
                    param_names.append(col_name)
                else:
                    bench_indices.append(i)
                    bench_names.append(col_name)
                    # Also add to our bench_columns if not already there
                    var found = False
                    for j in range(len(self.bench_columns)):
                        if self.bench_columns[j] == col_name:
                            found = True
                            break
                    if not found:
                        self.bench_columns.append(col_name)

            # Parse data rows
            for i in range(1, len(lines)):
                var line = lines[i].strip()
                if line == "":
                    continue

                var row = line.split(",")
                if len(row) < len(header):
                    continue

                var params = Dict[String, String]()
                for j in range(len(param_indices)):
                    params[param_names[j]] = String(
                        row[param_indices[j]].strip()
                    )

                for j in range(len(bench_indices)):
                    var bench_name = bench_names[j]
                    var val_str = row[bench_indices[j]].strip()
                    if len(val_str) > 0:
                        try:
                            # Simple float parsing
                            var val = Float64(String(val_str))
                            self.add_result(params, bench_name, val)
                        except:
                            # Skip if not a valid number
                            pass

    fn save_csv(self, filepath: String, autosave: Bool = True) raises:
        """Save results to CSV with timestamp.

        Args:
            filepath: Base path for CSV file.
            autosave: If False, skip saving (useful for development).

        Example:
            runner.save_csv("benches/results/my_benchmark")
            # Saves to: benches/results/my_benchmark_<timestamp>.csv

            runner.save_csv("benches/results/my_benchmark", autosave=False)
            # Skips saving
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

        var results_dir = base_dir
        if autosave:
            results_dir = getenv("MOJO_BENCH_RESULTS_DIR", "benches/results")

        # Create directory if it doesn't exist
        from os import makedirs

        # Use date folder when we have a human-readable timestamp.
        var date_folder = ""
        var ts = self.timestamp
        if len(ts) >= 10 and ts[4] == "_" and ts[7] == "_":
            date_folder = ts[:10]

        var final_dir = results_dir
        if date_folder != "":
            final_dir = results_dir + "/" + date_folder

        makedirs(final_dir, exist_ok=True)
        # Add timestamp (now human-readable instead of Unix)
        var final_path = (
            final_dir
            + "/"
            + filename
            + ("_" + self.timestamp if autosave else "")
            + ".csv"
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
