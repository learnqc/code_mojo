import subprocess
import sys
import os

def run_benchmark():
    print("Running benchmark... (this may take a while)")
    cmd = ["pixi", "run", "mojo", "benches/bench_high_n_simd_vs_numpy.mojo"]
    # Capture both stdout and stderr (for potential warnings)
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Benchmark failed!")
        print(result.stderr)
        sys.exit(1)
        
    return result.stdout

def parse_output(output):
    lines = output.splitlines()
    data_rows = []
    header = None
    
    for line in lines:
        line = line.strip()
        if "n, Size," in line:
            header = [x.strip() for x in line.split(',')]
            continue
            
        # Check if line starts with a number (data row)
        if line and line[0].isdigit():
            parts = [x.strip() for x in line.split(',')]
            if len(parts) >= 7:
                data_rows.append(parts)
                
    return header, data_rows

def format_cell(val, col_idx):
    if col_idx == 1: # Size
        # Format size with scientific notation or commas?
        # Maybe just keep as is or commas
        try:
            return f"{int(val):,}"
        except:
            return val
    if col_idx >= 2: # ms or speedup
        try:
            f = float(val)
            if col_idx == 6: # Speedup
                return f"{f:.2f}x"
            else: # Time
                return f"{f:.2f}"
        except:
            return val
    return val

def generate_markdown(header, rows):
    md = "# Benchmark Report: Butterfly vs NumPy (High N)\n\n"
    md += "Comparison of Large Scale FFT implementations.\n\n"
    
    if not header or not rows:
        md += "No data found.\n"
        return md
        
    # Markdown Table
    # Header
    md_header = "| " + " | ".join(header) + " |"
    md_sep = "| " + " | ".join(["---"] * len(header)) + " |"
    
    md += md_header + "\n"
    md += md_sep + "\n"
    
    for row in rows:
        formatted_row = [format_cell(val, i) for i, val in enumerate(row)]
        md += "| " + " | ".join(formatted_row) + " |" + "\n"
        
    md += "\n## Analysis\n"
    avg_speedup = 0
    count = 0
    for row in rows:
        try:
            speedup = float(row[6])
            avg_speedup += speedup
            count += 1
        except:
            pass
            
    if count > 0:
        avg_speedup /= count
        md += f"**Average Butterfly Speedup vs NumPy**: {avg_speedup:.2f}x\n"
        
    return md

def main():
    output = run_benchmark()
    print("Benchmark completed. Parsing results...")
    
    header, rows = parse_output(output)
    report = generate_markdown(header, rows)
    
    output_path = "benches/scaling_results_high_n.md"
    with open(output_path, "w") as f:
        f.write(report)
        
    print(f"Report saved to {output_path}")
    print(report)

if __name__ == "__main__":
    main()
