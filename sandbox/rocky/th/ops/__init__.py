from .base_ops import is_cuda
from .conversion_ops import variable, as_variable, to_numpy, variable_from_numpy, variable_from_tensor
from .context_ops import get_scope_str, get_phase, INIT, TRAIN, TEST, scope, inc_scope, reset, get_variable, phase
from .registry import register, registering, wrap
from .init_ops import zeros_initializer, ones_initializer, uniform_initializer
from .functional_ops import l2_normalize, pad
from .nn_ops import *
from .optimizers import get_optimizer
