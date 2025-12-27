"""Execution strategy for quantum circuits."""

from utils import Variant


struct Generic(Copyable, Movable, Stringable):
    """Simple generic execution (debuggable)."""

    fn __init__(out self):
        pass

    fn __str__(self) -> String:
        return "Generic (debuggable)"


struct SIMD(Copyable, Movable, Stringable):
    """SIMD-optimized execution."""

    fn __init__(out self):
        pass

    fn __str__(self) -> String:
        return "SIMD optimized (runtime dispatch)"


struct SIMDv2(Copyable, Movable, Stringable):
    """SIMD v2 with optimized indexing."""

    fn __init__(out self):
        pass

    fn __str__(self) -> String:
        return "SIMD v2 with dispatch"


struct FusedV3(Copyable, Movable, Stringable):
    """Fusion optimization."""

    fn __init__(out self):
        pass

    fn __str__(self) -> String:
        return "Fusion optimization"


# Execution strategy variant
alias ExecutionStrategy = Variant[Generic, SIMD, SIMDv2, FusedV3]

# Convenience aliases for common strategies
alias GENERIC = ExecutionStrategy(Generic())
alias SIMD_STRATEGY = ExecutionStrategy(SIMD())
alias SIMD_V2 = ExecutionStrategy(SIMDv2())
alias FUSED_V3 = ExecutionStrategy(FusedV3())

alias EXECUTION_STRATEGIES = [GENERIC, SIMD_STRATEGY, SIMD_V2, FUSED_V3]


fn get_strategy_description(strategy: ExecutionStrategy) -> String:
    """Get the description string for any execution strategy.

    This is the single source of truth for strategy descriptions.
    When adding a new strategy, only update this function.

    Args:
        strategy: The execution strategy variant.

    Returns:
        Human-readable description of the strategy.
    """
    if strategy.isa[Generic]():
        return String(strategy[Generic])
    elif strategy.isa[SIMD]():
        return String(strategy[SIMD])
    elif strategy.isa[SIMDv2]():
        return String(strategy[SIMDv2])
    elif strategy.isa[FusedV3]():
        return String(strategy[FusedV3])
    else:
        return "Unknown strategy"


fn get_strategy_name(strategy: ExecutionStrategy) -> String:
    """Get the function name for any execution strategy.

    This is the single source of truth for strategy names.
    When adding a new strategy, only update this function.

    Args:
        strategy: The execution strategy variant.

    Returns:
        Function name for the strategy (e.g., "execute_simd_v2").
    """
    if strategy.isa[Generic]():
        return "execute"
    elif strategy.isa[SIMD]():
        return "execute_simd"
    elif strategy.isa[SIMDv2]():
        return "execute_simd_v2"
    elif strategy.isa[FusedV3]():
        return "execute_fused_v3"
    else:
        return "unknown"
