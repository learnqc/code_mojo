from butterfly.core.types import *
from butterfly.core.state import *

# from butterfly.core.gates import *

# GridState is now List[List[Amplitude]] as defined in core/state.mojo or types?
# core/state.mojo uses GridState but doesn't define it?
# Wait, core/state.mojo imports types.
# types.mojo doesn't define GridState.
# __init__.mojo defined GridState.
# I should define GridState in types.mojo or state.mojo.
# Let's put it in types.mojo or keep it here.
# core/state.mojo uses GridState in init_state_grid signature.
# So GridState MUST be defined in types.mojo or state.mojo.
# I missed that.
# I will add GridState to types.mojo.
