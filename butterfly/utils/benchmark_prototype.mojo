from utils.variant import Variant
from time import perf_counter_ns, sleep
import benchmark
from benchmark import keep, run, Unit
from butterfly.utils.benchmark_runner import create_runner


fn test1(s: Tuple[String, Int]) -> Int:
    sleep(0.2)
    return len(s[0] * s[1])


fn test2(s: Tuple[String, Int]) -> Int:
    sleep(0.25)
    return len(s[0] * s[1]) * 2


fn main() raises:
    # constants for names

    alias NAME = "test_function_call"
    alias DESCRIPTION = "Test function call performance"

    alias INPUT = "input"
    alias COUNT = "count"

    alias TEST1 = "test1"
    alias TEST2 = "test2"

    # define parameter and benchmark columns
    var param_cols = List[String](INPUT, COUNT)
    var bench_cols = List[String](TEST1, TEST2)

    # create runner
    var runner = create_runner(NAME, DESCRIPTION, param_cols, bench_cols)

    # define inputs
    var inputs = List[Tuple[String, Int]]()
    inputs.append(Tuple("abc", 1))
    inputs.append(Tuple("def", 2))
    inputs.append(Tuple("ghi", 3))

    for input in inputs:
        var params = Dict[String, String]()
        params[INPUT] = input[0]
        params[COUNT] = String(input[1])

        runner.add_perf_result(params, TEST1, test1, input)
        runner.add_perf_result(params, TEST2, test2, input)

    runner.print_table()

    # optionally save to csv
    # runner.save_csv(NAME)

    # run with the bencmark runner
    # python benches/run_benchmark_suite.py --suite butterfly/utils/benhmark_suite_prototype.json --all
