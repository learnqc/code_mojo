#!/usr/bin/env python3
"""
Generate markdown reports from benchmark CSV files (stdlib only).

Usage:
    python benches/generate_report.py results/2024_12_24/value_encoding_executors_*.csv
    python benches/generate_report.py results/*/cft_comparison_*.csv --output report.md
"""

import csv
import argparse
from pathlib import Path
from typing import List, Dict
from collections import defaultdict


def load_csv(csv_file: str) -> List[Dict]:
    """Load a single CSV file."""
    rows = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['source_file'] = Path(csv_file).name
            row['date'] = Path(csv_file).parent.name
            rows.append(row)
    return rows


def detect_columns(rows: List[Dict]) -> tuple:
    """Detect parameter and benchmark columns."""
    if not rows:
        return [], []
    
    exclude = {'timestamp', 'source_file', 'date'}
    all_cols = [k for k in rows[0].keys() if k not in exclude]
    
    param_cols = []
    bench_cols = []
    
    # Benchmark keywords - if column name contains these, it's a benchmark
    benchmark_keywords = ['execute', 'time', 'ms', 'opt', 'simd', 'qft', 'cft', 'fft', 'v4', 'v5', 'v6', 'qiskit', 'fftw']
    
    for col in all_cols:
        # Check if column name suggests it's a benchmark
        is_benchmark_name = any(kw in col.lower() for kw in benchmark_keywords)
        
        if is_benchmark_name:
            bench_cols.append(col)
        else:
            # It's a parameter (n, k, value, block_size, etc.)
            param_cols.append(col)
    
    return param_cols, bench_cols


def generate_markdown_table(rows: List[Dict], param_cols: List[str], bench_cols: List[str]) -> str:
    """Generate a markdown table."""
    # Header
    header = "| " + " | ".join(param_cols) + " | " + " | ".join(bench_cols) + " | Winner |\n"
    separator = "|" + "|".join(["---"] * (len(param_cols) + len(bench_cols) + 1)) + "|\n"
    
    table_rows = []
    for row in rows:
        # Parameters
        param_vals = [row.get(col, "-") for col in param_cols]
        
        # Find winner
        bench_times = {}
        for col in bench_cols:
            try:
                bench_times[col] = float(row.get(col, 'inf'))
            except (ValueError, TypeError):
                pass
        
        if bench_times:
            winner = min(bench_times, key=bench_times.get)
            min_time = bench_times[winner]
            max_time = max(bench_times.values())
            speedup = max_time / min_time if min_time > 0 else 1.0
        else:
            winner = None
            speedup = 1.0
        
        # Benchmarks with winner marking
        bench_vals = []
        for col in bench_cols:
            try:
                val = float(row.get(col, 0))
                val_str = f"{val:.2f}"
                if col == winner:
                    val_str = f"**{val_str}***"
                bench_vals.append(val_str)
            except (ValueError, TypeError):
                bench_vals.append("-")
        
        # Winner
        if winner and speedup > 1.0:
            winner_text = f"{winner} ({speedup:.2f}x)"
        elif winner:
            winner_text = winner
        else:
            winner_text = "-"
        
        row_str = "| " + " | ".join(param_vals + bench_vals + [winner_text]) + " |"
        table_rows.append(row_str)
    
    return header + separator + "\n".join(table_rows)


def calculate_stats(rows: List[Dict], col: str) -> Dict:
    """Calculate statistics for a column."""
    vals = []
    for row in rows:
        try:
            vals.append(float(row.get(col, 0)))
        except (ValueError, TypeError):
            pass
    
    if not vals:
        return {}
    
    return {
        'mean': sum(vals) / len(vals),
        'min': min(vals),
        'max': max(vals),
        'std': (sum((x - sum(vals)/len(vals))**2 for x in vals) / len(vals)) ** 0.5
    }


def generate_report(csv_files: List[str], output_file: str = None):
    """Generate a markdown report from CSV files."""
    # Load all CSVs
    all_rows = []
    for f in csv_files:
        all_rows.extend(load_csv(f))
    
    if not all_rows:
        print("No data found!")
        return
    
    # Detect columns
    param_cols, bench_cols = detect_columns(all_rows)
    
    # Generate markdown
    md = []
    md.append("# Benchmark Report\n")
    
    # Metadata
    import time
    md.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    dates = set(row.get('date', 'unknown') for row in all_rows)
    md.append(f"**Date(s):** {', '.join(sorted(dates))}\n")
    md.append(f"**Total runs:** {len(all_rows)}\n")
    md.append(f"**Parameters:** {', '.join(param_cols)}\n")
    md.append(f"**Benchmarks:** {', '.join(bench_cols)}\n")
    md.append("\n## Results\n\n")  # Extra newline before table
    
    # Table
    table = generate_markdown_table(all_rows, param_cols, bench_cols)
    md.append(table)
    
    md.append("\n\n### Legend\n")
    md.append("- **Bold*** = Fastest for this parameter combination\n")
    md.append("- Winner column shows speedup vs slowest\n")
    
    # Summary statistics
    md.append("\n## Summary Statistics\n")
    for bench in bench_cols:
        stats = calculate_stats(all_rows, bench)
        if stats:
            md.append(f"\n### {bench}\n")
            md.append(f"- Mean: {stats['mean']:.2f} ms\n")
            md.append(f"- Min: {stats['min']:.2f} ms\n")
            md.append(f"- Max: {stats['max']:.2f} ms\n")
            md.append(f"- Std: {stats['std']:.2f} ms\n")
    
    # Write output
    report = "".join(md)
    
    if output_file:
        # Add timestamp to filename if not already present
        output_path = Path(output_file)
        # if '_' not in output_path.stem or not output_path.stem.split('_')[-1].isdigit():
        #     # Human-readable format: YYYY_MM_DD_HHMMSS
        #     timestamp = time.strftime('%Y_%m_%d_%H%M%S')
        #     new_name = f"{output_path.stem}_{timestamp}{output_path.suffix}"
        #     output_path = output_path.parent / new_name
        
        output_path.write_text(report)
        print(f"Report saved to: {output_path}")
    else:
        print(report)


def main():
    parser = argparse.ArgumentParser(description="Generate markdown reports from benchmark CSVs")
    parser.add_argument("csv_files", nargs="+", help="CSV files to process")
    parser.add_argument("-o", "--output", help="Output markdown file (default: print to stdout)")
    
    args = parser.parse_args()
    
    # Expand globs
    csv_files = []
    for pattern in args.csv_files:
        matches = list(Path(".").glob(pattern))
        if matches:
            csv_files.extend(str(p) for p in matches)
        else:
            # Try as literal path
            if Path(pattern).exists():
                csv_files.append(pattern)
    
    if not csv_files:
        print(f"No files found matching: {args.csv_files}")
        return
    
    print(f"Processing {len(csv_files)} file(s)...")
    generate_report(csv_files, args.output)


if __name__ == "__main__":
    main()
