import random
import time
import math
from math import sqrt, cos, sin, log2, log10, atan2, floor
from complex import ComplexSIMD
from testing import assert_true, assert_almost_equal

from butterfly import *
from butterfly.core.gates import *

def init_state(n: UInt) -> State:
    var state:State = [`1` if i == 0 else `0` for i in range(2 ** n)]
    return state^

def init_state_a[n: UInt]() -> ArrayState[1<<n]:
    var state = ArrayState[1 << n](fill=`0`)
    state[0] = `1`
    return state^

def generate_state(n:UInt, seed: Int = 555) -> State:
    # Choose a seed
    random.seed(seed)
    # Generate random probabilities that add up to 1
    var probs: List[FloatType] = [abs(random.random_float64(0, 1).cast[Type]()) for _ in range(2**n)]

    total: FloatType = 0.0
    for p in probs:
        total += p

    for i in range(len(probs)):
        probs[i] = probs[i]/total

    # Generate random angles in radians
    angles = [random.random_float64(0, 2 * math.pi).cast[Type]() for _ in range(2**n)]
    # Build the quantum state array
    state = [sqrt(p)*cis(theta) for (p, theta) in zip(probs, angles)]
    return state^

fn process_pair(mut state: State, gate: Gate, k0: UInt, k1: UInt):
    x = state[k0]
    y = state[k1]
    # new amplitudes
    state[k0] = x * gate[0][0] + y * gate[0][1]
    state[k1] = x * gate[1][0] + y * gate[1][1]

fn transform(mut state: State, target: UInt, gate: Gate):
    l = len(state)
    stride = 1 << target
    r  = 0
    for j in range(l//2):
        idx = 2*j - r     # r = j%stride
        process_pair(state, gate, idx, idx + stride)

        r += 1
        if r == stride:
            r = 0

fn process_pair_a(mut state: ArrayState, gate: Gate, k0: UInt, k1: UInt):
    x = state[k0]
    y = state[k1]
    # new amplitudes
    state[k0] = x * gate[0][0] + y * gate[0][1]
    state[k1] = x * gate[1][0] + y * gate[1][1]

fn transform_a(mut state: ArrayState, target: UInt, gate: Gate):
    l = len(state)
    stride = 1 << target
    r  = 0
    for j in range(l//2):
        idx = 2*j - r     # r = j%stride
        process_pair_a(state, gate, idx, idx + stride)

        r += 1
        if r == stride:
            r = 0

fn transform_swap(mut state: State, target: UInt, gate: Gate):
    l = len(state)
    stride = 1 << target
    # swap
    r  = 0
    for j in range(0, l//4):
        idx = 4*j - r
        state.swap_elements(idx + 1, idx + stride)

        r += 2
        if r >= stride:
            r = 0

#    apply gate to consecutive entries
    for j in range(0, l//2):
        idx = 2*j
        process_pair(state, gate, idx, idx + 1)

#   swap back
    r  = 0
    for j in range(0, l//4):
        idx = 4*j - r
        state.swap_elements(idx + 1, idx + stride)

        r += 2
        if r >= stride:
            r = 0

def is_bit_set(m: UInt, k: UInt) -> Bool:
    return m & UInt(1 << k) != 0

def c_transform(mut state: State, control: UInt, target: UInt, gate: Gate):
    stride = 1 << target
    r  = 0
    for j in range(len(state)//2):
        idx = 2*j - r     # r = j%stride
        if is_bit_set(UInt(idx), control):
            process_pair(state, gate, UInt(idx), UInt(idx + stride))

        r += 1
        if r == stride:
            r = 0

def iqft(mut state: State, targets: List[UInt]):
    for j in reversed(range(len(targets))):
        transform(state, targets[j], H)
        for k in reversed(range(j)):
            c_transform(state, targets[j], targets[k], P(-pi / 2 ** (j - k)))

def measure_qubit(mut state: State, t: UInt, reset:Bool, v: Bool) -> Bool:
    # n = UInt(log2(len(state)))

    stride = 1<<t

    prob0_sq: FloatType = 0
    prob1_sq: FloatType = 0

    r = 0
    for j in range(len(state)//2):
        idx = 2*j - r     # r = j%stride
        prob0_sq += state[idx].re*state[idx].re + state[idx].im*state[idx].im
        prob1_sq += state[idx + stride].re*state[idx + stride].re + state[idx + stride].im*state[idx + stride].im

        r += 1
        if r == stride:
            r = 0


    if v == False:
        r = 0
        for j in range(len(state)//2):
            idx = 2*j - r     # r = j%stride
            state[idx].re /= sqrt(prob0_sq)
            state[idx].im /= sqrt(prob0_sq)
            state[idx + stride] = `0`

            r += 1
            if r == stride:
                r = 0

    else:
        r = 0
        for j in range(len(state)//2):
            idx = 2*j - r     # r = j%stride
            state[idx] = `0`
            state[idx + stride].re /= sqrt(prob1_sq)
            state[idx + stride].im /= sqrt(prob1_sq)

            r += 1
            if r == stride:
                r = 0

        if reset:
            transform(state, t, X)

    return v




