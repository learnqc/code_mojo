"""
Flexible Hybrid SIMD Benchmark.
Comparing Library Grid (Best RowBits) vs Hybrid Grid (Best RowBits).
Target: Beat the 226ms @ n=24 4-row scalar record.
"""

from butterfly.core.types import FloatType, Type
from butterfly.core.gates import Gate, H
from butterfly.core.grid_state import GridQuantumState
from butterfly.core.circuit import QuantumCircuit, GateTransformation
from butterfly.algos.value_encoding_circuit import (
    encode_value_circuits_runtime,
    encode_value_circuit,
)
from butterfly.utils.benchmark_runner import BenchmarkRunner
from buffer import NDBuffer
from time import perf_counter_ns
from collections import Dict, List
from algorithm import parallelize
from benchmark import keep
from butterfly.core.grid_state_hybrid import HybridGrid


fn bench_n[
    n: Int, row_bits: Int
](mut runner: BenchmarkRunner, value: FloatType, iters: Int) raises:
    print(
        "Benchmarking n =",
        n,
        "row_bits =",
        row_bits,
        "with",
        iters,
        "iterations...",
    )
    var circuits = encode_value_circuits_runtime(n, value)
    var params = Dict[String, String]()
    params["n"] = String(n)
    params["row_bits"] = String(row_bits)
    alias row_size = 1 << (n - row_bits)

    # 1. Library Grid
    var t0 = Int(perf_counter_ns())
    for _ in range(iters):
        var s = GridQuantumState(n, row_bits)
        s.execute[row_size](circuits[0])
        s.execute[row_size](circuits[1])
        s.execute[row_size](circuits[2])
        keep(s.re.unsafe_ptr())
    var t1 = Int(perf_counter_ns())
    runner.add_result(
        params, "lib_grid_ms", Float64(t1 - t0) / 1_000_000.0 / iters
    )

    # 2. Hybrid Pure
    var x = HybridGrid(n, row_bits)
    t0 = Int(perf_counter_ns())
    for _ in range(iters):
        x.execute[True](circuits[0])
        x.execute[True](circuits[1])
        x.execute[True](circuits[2])
        keep(x.re.unsafe_ptr())
    t1 = Int(perf_counter_ns())
    runner.add_result(
        params, "hybrid_pure_ms", Float64(t1 - t0) / 1_000_000.0 / iters
    )


fn main() raises:
    print("Optimization Proof: Target 226ms @ n=24")
    var runner = BenchmarkRunner("Grid Configuration Comparison")
    var param_cols = List[String]()
    param_cols.append("n")
    param_cols.append("row_bits")
    runner.set_param_columns(param_cols^)
    var bench_cols = List[String]()
    bench_cols.append("lib_grid_ms")
    bench_cols.append("hybrid_pure_ms")
    runner.set_bench_columns(bench_cols^)
    alias val = 4.7

    # Test n=24 with row_bits=1 (2 rows) and row_bits=2 (4 rows)
    bench_n[24, 1](runner, val, 3)
    bench_n[24, 2](runner, val, 3)

    # Test n=27 with row_bits=2 (4 rows) which was the winner in lib
    bench_n[27, 2](runner, val, 1)

    runner.print_table(show_winner=True)
    runner.save_csv("proof_config_match")
