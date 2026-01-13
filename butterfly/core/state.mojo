from bit.bit import bit_reverse
import time
import math
from math import sqrt, cos, sin, log2, log10, atan2, floor
from testing import assert_true, assert_almost_equal
from sys.info import simd_width_of
from algorithm import vectorize, parallelize
from buffer import NDBuffer

from butterfly.utils.common import cis, swap_bits_index, parallelize_with_threads
from butterfly.utils.context import ExecContext

from butterfly.core.types import *

alias simd_width = simd_width_of[Type]()
alias QuantumState = State
alias Signal = State

struct State(Copyable, ImplicitlyCopyable, Movable, Sized):
    var re: List[FloatType]
    var im: List[FloatType]
    var _re_buf: NDBuffer[Type, 1, MutAnyOrigin, 1]
    var _im_buf: NDBuffer[Type, 1, MutAnyOrigin, 1]
    var _buf_valid: Bool

    fn __init__(out self, n: Int):
        self.re = List[FloatType](length=1 << n, fill=0.0)
        self.im = List[FloatType](length=1 << n, fill=0.0)
        self.re[0] = 1.0
        self.im[0] = 0.0
        self._re_buf = NDBuffer[Type, 1, MutAnyOrigin, 1](self.re)
        self._im_buf = NDBuffer[Type, 1, MutAnyOrigin, 1](self.im)
        self._buf_valid = True

    fn __init__(out self, var re: List[FloatType], var im: List[FloatType]):
        self.re = re^
        self.im = im^
        self._re_buf = NDBuffer[Type, 1, MutAnyOrigin, 1](self.re)
        self._im_buf = NDBuffer[Type, 1, MutAnyOrigin, 1](self.im)
        self._buf_valid = True

    fn __copyinit__(out self, existing: Self):
        self.re = List[FloatType](capacity=len(existing.re))
        self.im = List[FloatType](capacity=len(existing.im))
        for i in range(len(existing.re)):
            self.re.append(existing.re[i])
            self.im.append(existing.im[i])
        self._re_buf = NDBuffer[Type, 1, MutAnyOrigin, 1](self.re)
        self._im_buf = NDBuffer[Type, 1, MutAnyOrigin, 1](self.im)
        self._buf_valid = True

    fn __moveinit__(out self, deinit existing: Self):
        self.re = existing.re^
        self.im = existing.im^
        self._re_buf = NDBuffer[Type, 1, MutAnyOrigin, 1](self.re)
        self._im_buf = NDBuffer[Type, 1, MutAnyOrigin, 1](self.im)
        self._buf_valid = True

    fn __len__(self) -> Int:
        return self.size()

    fn size(self) -> Int:
        return len(self.re)

    fn __getitem__(self, idx: Int) -> Amplitude:
        return Amplitude(self.re[idx], self.im[idx])

    fn __setitem__(mut self, idx: Int, val: Amplitude):
        self.re[idx] = val.re
        self.im[idx] = val.im

    fn __iter__(self) -> _StateIterator:
        """Return an iterator over the amplitudes."""
        return _StateIterator(self)

    fn _ensure_buffers(mut self):
        if self._buf_valid:
            return
        self._re_buf = NDBuffer[Type, 1, _, 1](self.re)
        self._im_buf = NDBuffer[Type, 1, _, 1](self.im)
        self._buf_valid = True

    fn invalidate_buffers(mut self):
        self._buf_valid = False

    fn re_buffer(mut self) -> NDBuffer[Type, 1, MutAnyOrigin, 1]:
        self._ensure_buffers()
        return self._re_buf

    fn im_buffer(mut self) -> NDBuffer[Type, 1, MutAnyOrigin, 1]:
        self._ensure_buffers()
        return self._im_buf

    fn re_ptr(mut self) -> UnsafePointer[FloatType, MutAnyOrigin]:
        return self.re.unsafe_ptr()

    fn im_ptr(mut self) -> UnsafePointer[FloatType, MutAnyOrigin]:
        return self.im.unsafe_ptr()


struct _StateIterator(Copyable, ImplicitlyCopyable):
    """Iterator for State that yields Amplitude values."""

    var state: State
    var index: Int

    fn __init__(out self, state: State):
        self.state = state.copy()
        self.index = 0

    fn __copyinit__(out self, existing: Self):
        self.state = existing.state.copy()
        self.index = existing.index

    fn __iter__(self) -> Self:
        return self

    fn __has_next__(self) -> Bool:
        return self.index < self.state.size()

    fn __next__(mut self) -> Amplitude:
        var result = self.state[self.index]
        self.index += 1
        return result

    fn __len__(self) -> Int:
        """Return remaining elements."""
        return self.state.size() - self.index


fn bit_reverse_state(
    mut state: State, ctx: ExecContext = ExecContext()
):
    var n = state.size()
    if n <= 0 or (n & (n - 1)) != 0:
        return
    var log_n = 0
    var tmp = n
    while tmp > 1:
        tmp >>= 1
        log_n += 1

    var s_re = List[FloatType](length=n, fill=0.0)
    var s_im = List[FloatType](length=n, fill=0.0)

    if ctx.threads >= 0:
        # Optimized Parallel SIMD
        var ptr_in_re = state.re.unsafe_ptr()
        var ptr_in_im = state.im.unsafe_ptr()
        var ptr_out_re = s_re.unsafe_ptr()
        var ptr_out_im = s_im.unsafe_ptr()

        @parameter
        fn worker(idx: Int):
            alias width = simd_width
            var base = idx * width

            # Vectorized Index Generation
            var offsets = SIMD[DType.uint64, width]()
            for i in range(width):
                offsets[i] = i
            var vec_idx = SIMD[DType.uint64, width](base) + offsets

            # Bit Reverse
            var r_idx_u64 = bit_reverse(vec_idx) >> (64 - log_n)
            var r_idx = r_idx_u64.cast[DType.int64]()

            var val_re = ptr_in_re.gather(r_idx)
            var val_im = ptr_in_im.gather(r_idx)

            ptr_out_re.store(base, val_re)
            ptr_out_im.store(base, val_im)

        var vec_count = n // simd_width
        var threads = ctx.threads
        if threads > 0:
            parallelize[worker](vec_count, threads)
        else:
            parallelize[worker](vec_count)
        var tail_start = vec_count * simd_width
        for i in range(tail_start, n):
            var r_idx = Int(
                bit_reverse(SIMD[DType.uint64, 1](i))[0] >> (64 - log_n)
            )
            ptr_out_re[i] = ptr_in_re[r_idx]
            ptr_out_im[i] = ptr_in_im[r_idx]
    else:
        # Sequential Scalar (Fallback)
        # Using unsafe pointers for performance even in sequential mode
        var ptr_in_re = state.re.unsafe_ptr()
        var ptr_in_im = state.im.unsafe_ptr()
        var ptr_out_re = s_re.unsafe_ptr()
        var ptr_out_im = s_im.unsafe_ptr()

        for i in range(n):
            var r_idx = Int(
                bit_reverse(SIMD[DType.uint64, 1](i))[0] >> (64 - log_n)
            )
            ptr_out_re[i] = ptr_in_re[r_idx]
            ptr_out_im[i] = ptr_in_im[r_idx]

    state.re = s_re^
    state.im = s_im^
    state.invalidate_buffers()

fn bit_reverse_state_ndbuffer(
    mut state: State, parallel: Bool = True, ctx: ExecContext = ExecContext()
):
    var n = state.size()
    if n <= 0 or (n & (n - 1)) != 0:
        return
    var log_n = 0
    var tmp = n
    while tmp > 1:
        tmp >>= 1
        log_n += 1

    var s_re = List[FloatType](length=n, fill=0.0)
    var s_im = List[FloatType](length=n, fill=0.0)

    var in_re = state.re_buffer()
    var in_im = state.im_buffer()
    var out_re = NDBuffer[Type, 1, _, 1](s_re)
    var out_im = NDBuffer[Type, 1, _, 1](s_im)

    if parallel:
        @parameter
        fn worker(i: Int):
            var r_idx = Int(
                bit_reverse(SIMD[DType.uint64, 1](i))[0] >> (64 - log_n)
            )
            out_re[i] = in_re[r_idx]
            out_im[i] = in_im[r_idx]

        var threads = ctx.threads
        if threads > 0:
            parallelize[worker](n, threads)
        else:
            parallelize[worker](n)
    else:
        for i in range(n):
            var r_idx = Int(
                bit_reverse(SIMD[DType.uint64, 1](i))[0] >> (64 - log_n)
            )
            out_re[i] = in_re[r_idx]
            out_im[i] = in_im[r_idx]

    state.re = s_re^
    state.im = s_im^
    state.invalidate_buffers()

fn partial_bit_reverse_state(mut state: QuantumState, targets: List[Int]):
    """
    Perform bit-reversal swapping on a subset of qubits.
    """
    var k = len(targets)
    if k <= 1:
        return

    # Sort targets ascending
    var sorted_targets = targets.copy()
    for i in range(k):
        for j in range(i + 1, k):
            if sorted_targets[i] > sorted_targets[j]:
                var temp = sorted_targets[i]
                sorted_targets[i] = sorted_targets[j]
                sorted_targets[j] = temp

    var size = state.size()
    var buf_re = state.re_buffer()
    var buf_im = state.im_buffer()

    for i_large in range(size):
        # Extract small index i from large index i_large
        var i_small = 0
        for b in range(k):
            if (i_large >> sorted_targets[b]) & 1:
                i_small |= 1 << b

        # Reverse i_small to get j_small
        var j_small = 0
        for b in range(k):
            if (i_small >> b) & 1:
                j_small |= 1 << (k - 1 - b)

        if i_small < j_small:
            # Construct j_large from i_large by replacing target bits
            var j_large = i_large
            for b in range(k):
                var bit_val = (j_small >> b) & 1
                if bit_val:
                    j_large |= 1 << sorted_targets[b]
                else:
                    j_large &= ~(1 << sorted_targets[b])

            var tmp_re = buf_re[i_large]
            var tmp_im = buf_im[i_large]
            buf_re[i_large] = buf_re[j_large]
            buf_im[i_large] = buf_im[j_large]
            buf_re[j_large] = tmp_re
            buf_im[j_large] = tmp_im


fn swap_bits_state(mut state: State, bit_a: Int, bit_b: Int):
    """Swap two bit positions across all indices in the state."""
    if bit_a == bit_b:
        return

    var n = state.size()
    if n <= 0 or (n & (n - 1)) != 0:
        return
    for i in range(n):
        var j = swap_bits_index(i, bit_a, bit_b)
        if j > i:
            var tmp = state[i]
            state[i] = state[j]
            state[j] = tmp


fn swap_end_bits_state(mut state: State, nbits: Int):
    """Swap the least- and most-significant bits across the state."""
    var n = state.size()
    if n <= 0 or (n & (n - 1)) != 0:
        return
    if nbits < 2:
        return
    swap_bits_state(state, 0, nbits - 1)


fn swap_low_bits_state(mut state: State):
    """Swap the two least-significant bits across the state (bit 0 <-> bit 1)."""
    var n = state.size()
    if n <= 0 or (n & (n - 1)) != 0:
        return
    if n < 4:
        return

    for base in range(0, n, 4):
        var i = base + 1
        var j = base + 2
        var tmp = state[i]
        state[i] = state[j]
        state[j] = tmp


fn swap_high_bits_state(mut state: State, nbits: Int):
    """Swap the two most-significant bits across the state."""
    var n = state.size()
    if n <= 0 or (n & (n - 1)) != 0:
        return
    if nbits < 2:
        return
    swap_bits_state(state, nbits - 1, nbits - 2)

fn generate_state(n: Int, seed: Int = 555) -> State:
    import random
    random.seed(seed)
    var probs = List[FloatType](capacity=1 << n)
    for _ in range(1 << n):
        probs.append(abs(random.random_float64(0, 1).cast[Type]()))

    total: FloatType = 0.0
    for p in probs:
        total += p

    for i in range(len(probs)):
        probs[i] = probs[i] / total

    var state = QuantumState(n)
    for i in range(1 << n):
        theta = random.random_float64(0, 2 * math.pi).cast[Type]()
        state[i] = sqrt(probs[i]) * cis(theta)

    return state^

from butterfly.utils.config_global import get_global_config_int
from time import perf_counter_ns
from math import sqrt
from butterfly.core.types import FloatType
from collections import List


fn apply_bit_reverse(
    mut state: QuantumState,
    targets: List[Int],
) raises:
    if len(targets) == 0:
        bit_reverse_state(state)
    else:
        partial_bit_reverse_state(state, targets)


fn apply_permute_qubits(
    mut state: QuantumState,
    targets: List[Int],
) raises:
    var n = len(targets)
    if n == 0:
        return
    var tmp = state.size()
    var nbits = 0
    while tmp > 1:
        tmp //= 2
        nbits += 1
    if n != nbits:
        raise Error("Permutation size mismatch: " + String(n))
    var seen = List[Bool](length=n, fill=False)
    for i in range(n):
        var v = targets[i]
        if v < 0 or v >= nbits:
            raise Error("Invalid permutation value: " + String(v))
        if seen[v]:
            raise Error("Duplicate permutation value: " + String(v))
        seen[v] = True

    var perm = List[Int](capacity=n)
    for i in range(n):
        perm.append(i)

    for i in range(n):
        if perm[i] == targets[i]:
            continue
        var j = i + 1
        while j < n and perm[j] != targets[i]:
            j += 1
        if j >= n:
            raise Error("Invalid permutation order")
        swap_bits_state(state, i, j)
        var tmp = perm[i]
        perm[i] = perm[j]
        perm[j] = tmp


fn apply_swap(
    mut state: QuantumState,
    targets: List[Int],
) raises:
    if len(targets) != 2:
        raise Error("SWAP expects 2 targets")
    var a = targets[0]
    var b = targets[1]
    if a < 0 or a >= state.size() or b < 0 or b >= state.size():
        raise Error("Target out of bounds: " + String(a) + "," + String(b))
    swap_bits_state(state, a, b)


fn apply_measure(
    mut state: QuantumState,
    targets: List[Int],
    values_in: List[Optional[Bool]],
    seed_in: Optional[Int],
) raises:
    if len(targets) == 0:
        return
    var values = values_in.copy()
    if len(values) == 0:
        values = List[Optional[Bool]](
            length=len(targets), fill=None
        )
    if len(values) != len(targets):
        raise Error("Measurement values size mismatch")
    for t in targets:
        if t < 0 or t >= state.size():
            raise Error("Target out of bounds: " + String(t))

    var free_targets = List[Int]()
    var free_pos = List[Int](capacity=len(values))
    for j in range(len(values)):
        if values[j]:
            free_pos.append(-1)
        else:
            free_pos.append(len(free_targets))
            free_targets.append(targets[j])

    var size = state.size()
    var outcome_count = 1 << len(free_targets)
    var outcome_probs = List[FloatType](
        length=outcome_count, fill=0.0
    )
    var total = FloatType(0.0)

    for i in range(size):
        var is_match = True
        for j in range(len(targets)):
            var opt_val = values[j]
            if opt_val:
                var bit = (i >> targets[j]) & 1
                var desired = 1 if opt_val.value() else 0
                if bit != desired:
                    is_match = False
                    break
        if is_match:
            var outcome = 0
            for k in range(len(free_targets)):
                if (i >> free_targets[k]) & 1:
                    outcome |= 1 << k
            var re = FloatType(state.re[i])
            var im = FloatType(state.im[i])
            var p = re * re + im * im
            outcome_probs[outcome] = outcome_probs[outcome] + p
            total += p

    if total <= 0:
        raise Error("Measurement probability is zero")

    var seed = (
        seed_in.value()
        if seed_in
        else get_global_config_int("measure_seed", -1)
    )
    if seed < 0:
        seed = Int(perf_counter_ns())
        if seed < 0:
            seed = -seed
    var rng = seed % 2147483647
    rng = (rng * 48271) % 2147483647
    var r = FloatType(rng) / FloatType(2147483647) * total

    var chosen = 0
    var acc = FloatType(0.0)
    for i in range(outcome_count):
        acc += outcome_probs[i]
        if r <= acc:
            chosen = i
            break

    var chosen_prob = outcome_probs[chosen]
    if chosen_prob <= 0:
        raise Error("Measurement outcome probability is zero")

    var inv = FloatType(1.0) / FloatType(sqrt(chosen_prob))
    for i in range(size):
        var is_match = True
        for j in range(len(targets)):
            var bit = (i >> targets[j]) & 1
            var desired = 0
            var opt_val = values[j]
            if opt_val:
                desired = 1 if opt_val.value() else 0
            else:
                var pos = free_pos[j]
                desired = (chosen >> pos) & 1
            if bit != desired:
                is_match = False
                break
        if is_match:
            state.re[i] = FloatType(state.re[i]) * inv
            state.im[i] = FloatType(state.im[i]) * inv
        else:
            state.re[i] = FloatType(0.0)
            state.im[i] = FloatType(0.0)
    state.invalidate_buffers()
