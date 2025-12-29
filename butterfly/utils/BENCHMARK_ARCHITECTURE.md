# Benchmark Architecture Guide

## Overview

This document describes the agnostic benchmarking infrastructure for the Butterfly project. The architecture is designed to handle benchmarks with minimal code duplication and maximum maintainability.

## Core Principles

1. **Domain-Blind Infrastructure**: The `BenchmarkRunner` has no knowledge of domain-specific types (e.g., `QuantumState`, FFT arrays). It only orchestrates verification and performance measurement.

2. **Injected Verification Hooks**: Custom comparison logic is provided via function parameters, keeping the runner agnostic.

3. **Strict Separation of Concerns**: 
   - Generic utilities in `butterfly/utils/`
   - Domain-specific logic in `butterfly/core/` or benchmark files
   - No cross-contamination

## Architecture Layers

### Layer 1: Generic Interop (`python_interop.mojo`)

**Purpose**: Domain-blind Python function calling.

**Key Functions**:
- `python_call(module, func, ...)` - Returns raw `PythonObject`
- `python_to_float64(obj)` - Principle 89 compliant conversion
- `_ensure_path()` - Handles `sys.path` injection

**Critical Rule**: This file MUST NOT import domain types like `QuantumState`.

### Layer 2: Domain Assemblers (`quantum_interop.mojo`)

**Purpose**: Convert raw Python data into domain-specific Mojo types.

**Key Functions**:
- `state_from_python(data, n)` - Assembles `QuantumState` from Python lists
- `get_qiskit_state(n, value)` - High-level Qiskit integration

**Pattern**: Uses Layer 1 utilities internally, bridges to domain types.

### Layer 3: Benchmark Runner (`benchmark_runner.mojo`)

**Purpose**: Agnostic orchestration of verification and performance measurement.

**Key Features**:
- Generic `verify()` with custom comparison hooks
- Performance measurement via `add_perf_result()`
- Table printing and CSV export
- Winner detection and formatting

## Usage Patterns

### Pattern 1: Basic Benchmark (Mojo-only)

```mojo
from butterfly.core.state import QuantumState
from butterfly.utils.benchmark_runner import create_runner
from collections import Dict, List

fn executor_v1(n: Int) raises -> QuantumState:
    return QuantumState(n)

fn executor_v2(n: Int) raises -> QuantumState:
    return QuantumState(n)

fn compare_states(s1: QuantumState, s2: QuantumState, tol: Float64) raises:
    # Custom comparison logic
    if s1.size() != s2.size(): raise Error("Size mismatch")
    # ... more checks ...

fn main() raises:
    var param_cols = List[String]("n")
    var bench_cols = List[String]("v1", "v2")
    var runner = create_runner("my_benchmark", "Description", param_cols, bench_cols)

    for n in range(3, 6):
        # Verify correctness first
        runner.verify(n, executor_v1, executor_v2, compare=compare_states)
        
        # Then measure performance
        var params = Dict[String, String]()
        params["n"] = String(n)
        runner.add_perf_result(params, "v1", executor_v1, n)
        runner.add_perf_result(params, "v2", executor_v2, n)

    runner.print_table()
```

### Pattern 2: Python Interop Benchmark

```mojo
from butterfly.core.state import QuantumState
from butterfly.core.quantum_interop import get_qiskit_state
from butterfly.utils.benchmark_runner import create_runner
from collections import Dict, List

fn executor_mojo(n: Int) raises -> QuantumState:
    return QuantumState(n)

fn executor_python(n: Int) raises -> QuantumState:
    return get_qiskit_state(n, 0.0)  # Domain assembler handles conversion

fn compare_states(s1: QuantumState, s2: QuantumState, tol: Float64) raises:
    # Same comparison logic as Pattern 1
    pass

fn main() raises:
    var param_cols = List[String]("n")
    var bench_cols = List[String]("mojo", "python")
    var runner = create_runner("mojo_vs_python", "Comparison", param_cols, bench_cols)

    var n = 4
    runner.verify(n, executor_mojo, executor_python, compare=compare_states)
    
    var params = Dict[String, String]()
    params["n"] = String(n)
    runner.add_perf_result(params, "mojo", executor_mojo, n)
    runner.add_perf_result(params, "python", executor_python, n)

    runner.print_table()
```

## Advanced Features

### CSV Export

Save results for later analysis or aggregation:

```mojo
fn main() raises:
    var runner = create_runner("my_benchmark", "Description", param_cols, bench_cols)
    
    # ... run benchmarks ...
    
    runner.print_table()
    runner.save_csv("my_benchmark")  # Saves to results/YYYY-MM-DD/my_benchmark.csv
```

### Benchmark Suite Integration

Run benchmarks via the centralized suite runner:

```bash
# Run all benchmarks in a suite
python benches/run_benchmark_suite.py --suite butterfly/utils/benchmark_suite_prototype.json --all

# Run specific benchmark
python benches/run_benchmark_suite.py --suite butterfly/utils/benchmark_suite_prototype.json --name function_calling
```

**Suite JSON Format**:
```json
{
  "benchmarks": [
    {
      "name": "function_calling",
      "path": "butterfly/utils/benchmark_prototype.mojo",
      "description": "Agnostic verification prototype",
      "estimated_time": "~5 seconds"
    }
  ]
}
```

## Reference Implementations

### `benchmark_prototype.mojo`
Demonstrates the basic pattern with two Mojo executors and a custom verification hook. Shows:
- Hook-based verification
- Performance measurement
- Table output
- CSV export (commented)
- Suite integration (commented)

### `benchmark_prototype_python_interop.mojo`
Demonstrates Python interop with the decoupled architecture. Shows:
- Domain assembler usage (`get_qiskit_state`)
- Generic interop layer (used internally)
- Bit-perfect verification between Mojo and Python

## Migration Guide (for existing benchmarks)

**Before** (one-off script):
```mojo
fn main():
    # 50+ lines of custom verification
    # 30+ lines of custom timing
    # 20+ lines of custom table printing
    # No reusability
```

**After** (agnostic pattern):
```mojo
fn main() raises:
    var runner = create_runner(...)
    runner.verify(...)
    runner.add_perf_result(...)
    runner.print_table()
    # ~10 lines total
```

**Steps**:
1. Identify executor functions (what produces the result)
2. Extract comparison logic into a hook function
3. Replace custom timing with `runner.add_perf_result()`
4. Replace custom printing with `runner.print_table()`
5. Add to benchmark suite JSON

## Best Practices

1. **Always verify before benchmarking**: Use `runner.verify()` to ensure correctness
2. **Use explicit tolerance**: Don't rely on default floating-point comparison
3. **Name executors clearly**: `executor_mojo`, `executor_python`, `executor_v1`, etc.
4. **Keep comparison hooks simple**: Focus on the domain-specific logic only
5. **Document expected behavior**: Add comments explaining what each executor does

## Troubleshooting

### "Verification failed: States differ"
- Check that both executors use the same input parameters
- Verify tolerance is appropriate for your domain
- Add debug prints in the comparison hook to see actual differences

### "defining 'main' within a package is not yet supported"
- This is a Mojo LSP warning, safe to ignore
- The code runs correctly via `pixi run mojo run -I .`

### Python import errors
- Ensure `sys.path` injection is working (handled by `python_interop.mojo`)
- Check that Python modules are in the correct location
- Verify `benches/` directory structure

## Future Enhancements

- [ ] Automatic statistical analysis (mean, stddev, confidence intervals)
- [ ] Parallel benchmark execution
- [ ] HTML report generation
- [ ] Integration with CI/CD pipelines
- [ ] Cross-platform result aggregation
