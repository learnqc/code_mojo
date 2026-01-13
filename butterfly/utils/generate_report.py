#!/usr/bin/env python3
"""
Generate markdown reports from benchmark CSV files (stdlib only).

Usage:
    python butterfly/utils/generate_report.py results/2024_12_24/value_encoding_executors_*.csv
    python butterfly/utils/generate_report.py results/*/cft_comparison_*.csv --output report.md
"""

import csv
import argparse
import time
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


def detect_columns(rows: List[Dict], header: List[str]) -> tuple:
    """Detect parameter and benchmark columns using header order."""
    if not rows:
        return [], []

    exclude = {'', 'timestamp', 'source_file', 'date'}
    all_cols = [k for k in header if k not in exclude]

    param_cols = []
    bench_cols = []

    common_params = {
        'n',
        'k',
        'm',
        'size',
        'len',
        'length',
        'depth',
        'width',
        'height',
        'batch',
        'batches',
        'block',
        'block_size',
        'threads',
        'cores',
        'samples',
        'reps',
        'iters',
        'trial',
        'trials',
    }

    for col in all_cols:
        numeric = True
        for row in rows:
            val = row.get(col, "")
            if val is None or val == "":
                continue
            try:
                float(val)
            except (ValueError, TypeError):
                numeric = False
                break
        if not numeric or col.lower() in common_params:
            param_cols.append(col)
        else:
            bench_cols.append(col)

    # If everything is numeric, treat the first column as a parameter.
    if not param_cols and all_cols:
        param_cols.append(all_cols[0])
        bench_cols = all_cols[1:]

    return param_cols, bench_cols


def generate_markdown_table(
    rows: List[Dict],
    param_cols: List[str],
    bench_cols: List[str],
    baseline: str = None,
) -> str:
    """Generate a markdown table."""
    # Header
    last_col = "Winner"
    if baseline:
        last_col = f"Winner (vs {baseline})"
    header = "|"
    if param_cols:
        header += " " + " | ".join(param_cols) + " |"
    if bench_cols:
        header += " " + " | ".join(bench_cols) + " |"
    header += " " + last_col + " |\n"
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
                    val_str = f"**{val_str}**"
                bench_vals.append(val_str)
            except (ValueError, TypeError):
                bench_vals.append("-")
        
        # Winner
        if winner:
            if baseline and baseline in bench_times and min_time > 0:
                base_time = bench_times[baseline]
                base_speedup = base_time / min_time if base_time > 0 else 1.0
                winner_text = f"{winner} ({base_speedup:.2f}x)"
            elif speedup > 1.0:
                winner_text = f"{winner} ({speedup:.2f}x)"
            else:
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


def _default_report_path(dates: List[str]) -> Path:
    """Build a default report path under benches/reports/<date>/."""
    report_date = "mixed"
    uniq = sorted(set(dates))
    if len(uniq) == 1 and uniq[0]:
        report_date = uniq[0]

    ts = time.strftime("%Y_%m_%d__%H_%M_%S")
    out_dir = Path("benches") / "reports" / report_date
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"report_{ts}.md"


def generate_report(
    csv_files: List[str],
    output_file: str = None,
    group_by: str = "none",
    baseline: str = None,
    sort_by: List[str] = None,
):
    """Generate a markdown report from CSV files."""
    # Load all CSVs
    all_rows = []
    header = []
    for f in csv_files:
        with open(f, 'r') as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames and not header:
                header = list(reader.fieldnames)
        all_rows.extend(load_csv(f))
    
    if not all_rows:
        print("No data found!")
        return
    
    # Detect columns
    if not header and all_rows:
        header = list(all_rows[0].keys())
    param_cols, bench_cols = detect_columns(all_rows, header)
    if baseline and baseline not in bench_cols:
        baseline = None
    
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

    if sort_by:
        def sort_key(row: Dict) -> tuple:
            vals = []
            for key in sort_by:
                val = row.get(key, "")
                try:
                    vals.append(float(val))
                except (ValueError, TypeError):
                    vals.append(val)
            return tuple(vals)
        all_rows = sorted(all_rows, key=sort_key)

    if group_by in ("date", "source_file"):
        groups = defaultdict(list)
        for row in all_rows:
            groups[row.get(group_by, "unknown")].append(row)
        for group_key in sorted(groups.keys()):
            md.append(f"### {group_by}: {group_key}\n\n")
            table = generate_markdown_table(
                groups[group_key], param_cols, bench_cols, baseline
            )
            md.append(table)
            md.append("\n\n")
    else:
        table = generate_markdown_table(all_rows, param_cols, bench_cols, baseline)
        md.append(table)
    
    md.append("\n\n### Legend\n")
    md.append("- **bold** = Fastest for this parameter combination\n")
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
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = _default_report_path(list(dates))

    output_path.write_text(report)
    print(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate markdown reports from benchmark CSVs")
    parser.add_argument("csv_files", nargs="+", help="CSV files or folders to process")
    parser.add_argument(
        "-o",
        "--output",
        help="Output markdown file (default: benches/reports/<date>/report_<timestamp>.md)",
    )
    parser.add_argument("--group-by", choices=["none", "date", "source_file"], default="none")
    parser.add_argument("--baseline", help="Benchmark column to use for speedup")
    parser.add_argument("--sort-by", help="Comma-separated sort keys (param columns)")
    
    args = parser.parse_args()
    
    # Expand globs and folders
    csv_files = []
    for pattern in args.csv_files:
        p = Path(pattern)
        if p.is_dir():
            csv_files.extend(str(f) for f in p.rglob("*.csv"))
            continue

        matches = list(Path(".").glob(pattern))
        if matches:
            csv_files.extend(str(f) for f in matches)
        else:
            # Try as literal path
            if p.exists():
                csv_files.append(pattern)
    
    if not csv_files:
        print(f"No files found matching: {args.csv_files}")
        return
    
    
    print(f"Generating reports from {len(csv_files)} file(s):")
    for csv_file in csv_files:
        print(f"  - {csv_file}")
    print()
    sort_by = [s.strip() for s in args.sort_by.split(",")] if args.sort_by else None
    generate_report(csv_files, args.output, args.group_by, args.baseline, sort_by)


if __name__ == "__main__":
    main()
