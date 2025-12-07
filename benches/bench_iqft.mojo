import benchmark
from butterfly.core.state import QuantumState, iqft, generate_state
from butterfly.utils.visualization import print_state

alias n = 14

fn main() raises:
    print("Benchmarking IQFT N={n}...")

    # Initialize random_state once locally
    var random_state = generate_state(n)
#     print_state(random_state)

    @parameter
    fn bench_iqft_no_swap():
        try:
            var state = random_state
            iqft(state, [j for j in range(n)], swap=False)
        except:
            pass

    @parameter
    fn bench_iqft_with_swap():
        try:
            var state = random_state
            iqft(state, [j for j in range(n)], swap=True)
        except:
            pass

    var report_no_swap = benchmark.run[bench_iqft_no_swap](5, 100)
    print("No Swap:", report_no_swap.mean(benchmark.Unit.ms), "ms")

    var report_swap = benchmark.run[bench_iqft_with_swap](5, 100)
    print("With Swap:", report_swap.mean(benchmark.Unit.ms), "ms")
