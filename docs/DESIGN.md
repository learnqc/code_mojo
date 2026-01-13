# Design Overview

This document captures the core design concepts and execution strategies in
the Butterfly codebase. It focuses on the progression from functional kernels
to circuit-level transformations and higher-level execution strategies.

## Core Concepts

### State Representation
- `State`/`QuantumState` stores amplitudes as separate real/imag arrays.
- Helpers like `bit_reverse_state` and grid views operate on this layout.

### Functional Kernels (State Mutations)
- Lowest-level operations are functions that mutate `QuantumState` in place.
- Examples: scalar kernels, SIMD kernels, and specialized sparse kernels
  (e.g., H/H, H/P, P/P fused kernels).

### Transformations
- A `Transformation` describes *what* to do without executing it immediately.
- The primary variant is `GateTransformation`:
  - target qubit(s)
  - control qubit(s)
  - gate metadata (`GateInfo`)
- Additional variants handle swaps, unitary blocks, measurement, and classical
  transforms.

### Circuits
- A `Circuit` is an ordered list of `Transformation` items.
- Circuits can be composed, appended, and fused.
- Fusing rewrites multiple transformations into fewer operations, often using
  specialized kernels for adjacent pairs.

## Execution Strategies

Execution is controlled through an `ExecContext` strategy selector. The main
strategies fall into three buckets:

### Scalar / SIMD / Parallel SIMD
- Scalar is the simplest reference path.
- SIMD improves throughput with vectorized kernels.
- SIMD Parallel adds work partitioning for higher concurrency.

### Grid Execution
- The state is interpreted as a 2D grid to increase locality.
- Row-local kernels operate on a contiguous row slice.
- Grid parallel splits rows across workers.

### Fused Grid Variants
- Grid fusion combines adjacent, row-local gates to reduce passes.
- Specialized H/H, H/P, P/P, and CP/CP kernels are used when patterns match.

## Fusion Pipeline

Fusion is implemented as a transformation rewrite pass:
- Same-target fusion composes adjacent gates into a single unitary.
- Specialized pair fusion recognizes patterns (H/H, H/P, P/P, CP/CP shared control).
- The executor dispatches to specialized kernels when fused pairs are present.

## Value / Function Encoding

Algorithms like value encoding and function encoding build circuits from:
- Initialization (H over key/value registers)
- Phase encodings via P/CP/MCP gates
- IQFT on the value register

These workflows emphasize:
- Dense use of phase gates (good fusion candidates)
- Consistent register layout (key bits then value bits)

## Visualization

The visualization utilities provide:
- Tabular amplitude views (with magnitude/phase)
- Grid-like 2D views keyed by `col_bits`, matching the grid execution layout
- Optional binary labels for rows/columns to aid debugging

## Benchmarks

Benchmarks follow a consistent pattern:
- Define `LabeledFunction` instances for each strategy.
- Optionally verify correctness against a baseline.
- Run performance timings with a runner that produces tables and CSVs.

## Summary

The architecture emphasizes:
- Simple, mutation-based kernels at the bottom
- Declarative transformation lists in the middle
- Pluggable execution strategies at the top
- Fusion as a rewrite pass that targets common gate patterns
