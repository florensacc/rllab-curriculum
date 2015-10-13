# The implementation largely referenced TrajOpt
# Let's first use a collocation formulation 

import numpy as np

def solve(x0, uinit, sysdyn, cost_func, final_cost_func, xinit=None):
    Dx = len(x0)
    Du = uinit.shape[1]
    N = len(uinit)
    if xinit is None:
        xinit = np.tile(x0.reshape(1, -1), (N+1, 1))
    import ipdb; ipdb.set_trace()
