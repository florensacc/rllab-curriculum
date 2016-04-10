# 2016-04-08

Features:
- Upgraded Mujoco interface to accomodate v1.30

# 2016-04-09

- Fixed bug of TNPG (max_backtracks should be set to 1 instead of 0) 
- Neural network policies now use tanh nonlinearities by default
- Refactored interface for `rllab.sampler.parallel_sampler`. Extracted new module `rllab.sampler.stateful_pool` containing general parallelization utilities.
- Fixed numerous issues in tests causing too long to run.
