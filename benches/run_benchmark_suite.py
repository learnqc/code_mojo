#!/usr/bin/env python3
"""
Benchmark Suite Runner

Maintains a curated list of key benchmarks that can be run with one command.
Useful for cross-platform performance testing.

Usage:
    python benches/run_benchmark_suite.py --all
    python benches/run_benchmark_suite.py --list
    python benches/run_benchmark_suite.py --run gates
    python benches/run_benchmark_suite.py --run value_encoding v3_vs_v2
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

import json

# Load benchmark registry from JSON
def load_benchmarks(suite_file="benches/benchmarks.json"):
    """Load benchmarks from external JSON file.
    
    Args:
        suite_file: Path to the benchmark suite JSON file
    """
    benchmarks_path = Path(suite_file)
    with open(benchmarks_path, 'r') as f:
        return json.load(f)

BENCHMARKS = load_benchmarks()


def list_benchmarks():
    """Print all available benchmarks."""
    print("Available Benchmarks")
    print("=" * 80)
    print()
    
    for key, bench in BENCHMARKS.items():
        print(f"[{key}]")
        print(f"  Name: {bench['name']}")
        print(f"  Path: {bench['path']}")
        print(f"  Description: {bench['description']}")
        print(f"  Estimated Time: {bench['estimated_time']}")
        print()


def run_benchmark(key: str, verbose: bool = True):
    """Run a single benchmark."""
    if key not in BENCHMARKS:
        print(f"Error: Unknown benchmark '{key}'")
        print(f"Available benchmarks: {', '.join(BENCHMARKS.keys())}")
        return False
    
    bench = BENCHMARKS[key]
    
    if verbose:
        print("=" * 80)
        print(f"Running: {bench['name']}")
        print(f"Path: {bench['path']}")
        print(f"Estimated Time: {bench['estimated_time']}")
        print("=" * 80)
        print()
    
    # Check if file exists
    if not Path(bench['path']).exists():
        print(f"Error: Benchmark file not found: {bench['path']}")
        return False
    
    # Setup results directory by date (Principle #18)
    print("  [Setup] Organizing results by date...")
    date_str = datetime.now().strftime("%Y_%m_%d")
    results_dir = Path("benches/results") / date_str
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Pass results dir via environment variable to BenchmarkRunner.mojo
    env = os.environ.copy()
    env["MOJO_BENCH_RESULTS_DIR"] = str(results_dir)
    
    # Run the benchmark
    print(f"  [Execution] Starting Mojo benchmark: {bench['path']}")
    cmd = ["pixi", "run", "mojo", "run", "-I", ".", bench['path']]
    
    try:
        # Track files before run
        before_files = set(results_dir.glob("*.csv"))
        
        result = subprocess.run(cmd, check=True, env=env)
        
        # Find the new CSV
        print("  [Parsing] Looking for new results...")
        after_files = set(results_dir.glob("*.csv"))
        new_files = after_files - before_files
        
        if new_files:
            csv_path = sorted(list(new_files))[-1]  # Get latest
            
            # Generate markdown report
            # Use benchmark filename as report name
            # report_name = Path(bench['path']).stem
            report_name, _ = os.path.splitext(os.path.basename(csv_path))
            report_dir = Path("benches/reports")
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / f"{report_name}.md"
            
            print(f"  [Reporting] Generating report from {csv_path.name}...")
            report_cmd = [sys.executable, "benches/generate_report.py", str(csv_path), "-o", str(report_path)]
            subprocess.run(report_cmd, check=True)

        if verbose:
            print()
            print(f"✓ Benchmark '{key}' completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Benchmark '{key}' failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n✗ Benchmark '{key}' interrupted by user")
        return False


def run_all_benchmarks():
    """Run all benchmarks in sequence."""
    print("Running All Benchmarks")
    print("=" * 80)
    print()
    
    start_time = datetime.now()
    results = {}
    
    for key in BENCHMARKS.keys():
        success = run_benchmark(key, verbose=True)
        results[key] = success
        print()
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("=" * 80)
    print("Benchmark Suite Summary")
    print("=" * 80)
    print()
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for key, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {status}: {key}")
    
    print()
    print(f"Total: {passed}/{total} passed")
    print(f"Duration: {duration}")
    print()
    
    return all(results.values())


def main():
    parser = argparse.ArgumentParser(
        description="Run Butterfly benchmark suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--suite",
        default="benches/benchmarks.json",
        help="Path to benchmark suite JSON file (default: benches/benchmarks.json)",
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available benchmarks",
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmarks",
    )
    
    parser.add_argument(
        "--run",
        nargs="+",
        metavar="BENCHMARK",
        help="Run specific benchmark(s) by key",
    )
    
    args = parser.parse_args()
    
    # Reload benchmarks from specified suite
    global BENCHMARKS
    BENCHMARKS = load_benchmarks(args.suite)
    
    # Default to listing if no args
    if not (args.list or args.all or args.run):
        list_benchmarks()
        return
    
    if args.list:
        list_benchmarks()
        return
    
    if args.all:
        success = run_all_benchmarks()
        sys.exit(0 if success else 1)
    
    if args.run:
        all_success = True
        for key in args.run:
            success = run_benchmark(key)
            all_success = all_success and success
            if len(args.run) > 1:
                print()
        
        sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
