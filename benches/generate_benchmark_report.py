
import subprocess
import re
import sys
import os

def run_benchmark():
    print("Running benchmark... (this may take a while)")
    # Run the mojo benchmark command
    cmd = ["pixi", "run", "mojo", "run", "-I", ".", "benches/bench_parallel_vs_numpy.mojo"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Benchmark failed!")
        print(result.stderr)
        sys.exit(1)
        
    return result.stdout

def parse_table(output):
    # Extract the table part
    lines = output.splitlines()
    table_lines = []
    capture = False
    for line in lines:
        if "n    | Size" in line:
            capture = True
        if capture:
            table_lines.append(line)
            if line.strip() and "Scalar" not in line and "Parallel" not in line and "|" in line:
                # Keep accumulating rows
                pass
    
    # Prune trailing empty lines or non-table lines if any
    # Simple strategy: Just grab lines that look like table rows
    final_rows = []
    header_found = False
    separator_found = False
    
    for line in lines:
        if "n    | Size" in line:
            header_found = True
            final_rows.append(line)
            continue
        if header_found and "-----" in line:
            separator_found = True
            final_rows.append(line)
            continue
        if separator_found:
            # Check if line looks like a data row "  8 |"
            if re.match(r'\s*\d+\s*\|', line):
                final_rows.append(line)
    
    return "\n".join(final_rows)

def analyze_results(table_str):
    # Basic textual analysis for the conclusion
    rows = table_str.splitlines()[2:] # Skip header and separator
    
    small_scalar_wins = False
    large_parallel_wins = False # Any parallel
    numpy_crossover = 0
    best_large_algo = "Parallel"
    
    for row in rows:
        parts = row.split('|')
        if len(parts) < 8: continue # increased col count
        
        try:
            n = int(parts[0].strip())
            winner = parts[7].strip() # Index 7 now for "Best Mojo vs NumPy"
            
            if n <= 10 and "Scalar" in winner:
                small_scalar_wins = True
            if n >= 16 and ("Par" in winner or "ParSIMD" in winner or "NDBuf" in winner):
                large_parallel_wins = True
                if "ParSIMD" in winner:
                    best_large_algo = "Parallel SIMD"
                if "NDBuf" in winner:
                    best_large_algo = "Parallel NDBuffer"
            
            # Rough parsing of speedup
            speedup_str = parts[7].split('x')[0].strip()
            speedup = float(speedup_str)
            
            if ("Par" in winner or "ParSIMD" in winner) and speedup > 1.0 and numpy_crossover == 0:
                numpy_crossover = n
                
        except:
            continue
            
    return small_scalar_wins, large_parallel_wins, numpy_crossover, best_large_algo

def format_value(val_str):
    try:
        # Try to parse as float
        # Check if it has suffixes
        suffix = ""
        if "(" in val_str:
            parts = val_str.split("(")
            val_str = parts[0]
            suffix = " (" + parts[1]
        
        f = float(val_str.strip())
        return f"{f:.3f}{suffix}"
    except:
        return val_str

def reformat_table(table_str):
    lines = table_str.splitlines()
    formatted_lines = []
    
    # Process header
    if len(lines) > 0:
        formatted_lines.append(lines[0]) # Header
    if len(lines) > 1:
        formatted_lines.append(lines[1]) # Separator
        
    for line in lines[2:]:
        parts = line.split('|')
        if len(parts) < 2:
            formatted_lines.append(line)
            continue
            
        # parts indices: 0=n, 1=Size, 2=Scalar, 3=Par, 4=ParSIMD, 5=NDBuf, 6=NumPy, 7=Speedup
        new_parts = []
        for i, part in enumerate(parts):
            if i >= 2 and i <= 7: # Time cols + speedup
                new_parts.append(f" {format_value(part)} ".ljust(len(part)))
            else:
                new_parts.append(part)
        
        formatted_lines.append("|".join(new_parts))
        
    return "\n".join(formatted_lines)

def generate_markdown(table_content, analysis):
    small_wins, large_wins, crossover, best_large = analysis
    
    clean_table = reformat_table(table_content)
    
    md = "# Walkthrough: Benchmarking Classical FFT\n\n"
    md += "## 1. Implementations\n"
    md += "Location: [butterfly/core/classical_fft.mojo](butterfly/core/classical_fft.mojo)\n\n"
    md += "- **Scalar**: Baseline.\n"
    md += "- **SIMD**: Vectorized (slower due to strided access).\n"
    md += "- **Parallel**: Uses `algorithm.parallelize`.\n"
    md += "- **Parallel SIMD**: Uses `parallelize` + explicit `vectorize` with strided gather.\n\n"
    md += "## 2. Scalar Butterfly vs Parallel Butterfly vs NumPy\n"
    md += "We compared the best Mojo implementation (Scalar for small, Parallel/Par-SIMD for large) against NumPy.\n\n"
    
    md += clean_table + "\n\n"
    
    md += "### Conclusion\n"
    if small_wins:
        md += "- **Small N**: **Mojo Scalar Butterfly** is faster than NumPy.\n"
    else:
        md += "- **Small N**: NumPy is competitive or faster.\n"
        
    md += "- **Medium N**: NumPy typically dominates due to optimized C/Fortran backend.\n"
    
    if large_wins:
        md += f"- **Large N**: **Mojo {best_large} Butterfly** becomes consistently **faster than NumPy**.\n"
        if crossover > 0:
            md += f"  - **Crossover Point**: n={crossover}.\n"
    else:
        md += "- **Large N**: NumPy retains the lead, but Mojo Parallel closes the gap.\n"
        
    return md

def main():
    output = run_benchmark()
    print("Benchmark completed. Parsing results...")
    
    table = parse_table(output)
    analysis = analyze_results(table)
    report = generate_markdown(table, analysis)
    
    output_path = "benchmark_parallel_fft_results.md"
    with open(output_path, "w") as f:
        f.write(report)
        
    print(f"Report saved to {output_path}")

if __name__ == "__main__":
    main()
