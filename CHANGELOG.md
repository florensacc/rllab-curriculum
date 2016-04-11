# 2016-04-08

Features:
- Upgraded Mujoco interface to accomodate v1.30

# 2016-04-09

- Fixed bug of TNPG (max_backtracks should be set to 1 instead of 0) 
- Neural network policies now use tanh nonlinearities by default
- Refactored interface for `rllab.sampler.parallel_sampler`. Extracted new module `rllab.sampler.stateful_pool` containing general parallelization utilities.
- Fixed numerous issues in tests causing too long to run.
- Merged release branch onto master and removed the release branch, to avoid potential confusions.

# 2016-04-10

- Known issues:
  - TRPO does not work well with relu since the hessian is undefined at 0, causing NaN sometimes. This issue of Theano is tracked here: https://github.com/Theano/Theano/issues/4353). If relu must be used, try using `theano.tensor.maximum(x, 0.)` as opposed to `theano.tensor.nnet.relu`.

# 2016-04-11

- Added a method `truncate_paths` to the `rllab.sampler.parallel_sampler` module. This should be sufficient to replace the old configurable parameter `whole_paths` which has been removed during refactoring.
