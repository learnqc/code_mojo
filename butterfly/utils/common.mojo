from butterfly.core.types import Complex, FloatType
from algorithm import parallelize
from math import cos, sin

alias PARALLEL_SCHEDULE_CHUNKED = 0
alias PARALLEL_SCHEDULE_FLAT = 1

fn cis(theta: FloatType) -> Complex:
    return Complex(cos(theta), sin(theta))


fn swap_bits_index(i: Int, a: Int, b: Int) -> Int:
    """Swap two bit positions in an index."""
    if a == b:
        return i
    var bit_a = (i >> a) & 1
    var bit_b = (i >> b) & 1
    var x = bit_a ^ bit_b
    return i ^ ((x << a) | (x << b))

from os import getenv, setenv
from sys.info import num_physical_cores, num_logical_cores

fn detect_physical_cores() -> Int:
    """Detect number of physical CPU cores."""
    var cores = num_physical_cores()
    if cores > 0:
        return cores
    return 1  # Minimum safe fallback
    
fn detect_logical_cores() -> Int:
    """Detect number of logical CPU cores (hardware threads)."""
    var cached = getenv("BUTTERFLY_CORE_COUNT")
    if cached != "":
        try:
            return atol(cached)
        except:
            pass

    var cores = num_logical_cores()
    if cores > 0:
        _ = setenv("BUTTERFLY_CORE_COUNT", String(cores), 1)
        return cores

    var phys = detect_physical_cores()
    _ = setenv("BUTTERFLY_CORE_COUNT", String(phys), 1)
    return phys

@parameter
fn parallelize_with_threads[func: fn(Int) capturing -> None](count: Int, threads: Int):
    """Run parallelize with optional explicit thread count."""
    if threads > 0:
        parallelize[func](count, threads)
    else:
        parallelize[func](count)


@parameter
fn parallelize_blocks[
    func: fn(Int, Int) capturing -> None
](total_blocks: Int, threads: Int, schedule: Int):
    """Parallelize over blocks with either chunked or flat scheduling."""
    # TODO: This helper currently crashes in bench_transform_h; avoid use until fixed.
    if total_blocks <= 0:
        return

    var num_workers = threads if threads > 0 else total_blocks
    var actual_workers = min(num_workers, total_blocks)
    var blocks_per_worker = max(1, total_blocks // actual_workers)

    if schedule == PARALLEL_SCHEDULE_CHUNKED:
        @parameter
        fn worker_chunked(item_id: Int):
            var start_block = item_id * blocks_per_worker
            var end_block = (
                total_blocks if item_id
                == actual_workers - 1 else (item_id + 1) * blocks_per_worker
            )
            func(start_block, end_block)

        var work_items = actual_workers
        if threads > 0:
            parallelize[worker_chunked](work_items, threads)
        else:
            parallelize[worker_chunked](work_items)
    else:
        @parameter
        fn worker_flat(item_id: Int):
            func(item_id, item_id + 1)

        var work_items = total_blocks
        if threads > 0:
            parallelize[worker_flat](work_items, threads)
        else:
            parallelize[worker_flat](work_items)
