.. _implement_algo_advanced:

======================================
Implementing New Algorithms (Advanced)
======================================

In this section, we will anatomize the implementation of vanilla policy gradient
algorithm provided in the algorithm, available at :code:`rllab/algo/vpg.py`. It utilizes
many functionalities provided by the framework, which we describe below.


The :code:`BatchPolopt` Class
=============================

The :code:`VPG` class inherits from :code:`BatchPolopt`, which is an abstract
class inherited by algorithms with a common structure. The structure is as
follows:

- Initialize policy :math:`\pi` with parameter :math:`\theta_1`.

- Initialize the computational graph structure.

- For iteration :math:`k = 1, 2, \ldots`:

    - Sample N trajectories :math:`\tau_1`, ..., :math:`\tau_n` under the
      current policy :math:`\theta_k`, where
      :math:`\tau_i = (s_t^i, a_t^i, R_t^i)_{t=0}^{T-1}`. Note that the last
      state is dropped since no action is taken after observing the last state.

    - Update the policy based on the collected on-policy trajectories.

    - Print diagnostic information and store intermediate results.

Note the parallel between the structure above and the pseudocode for VPG. The
:code:`BatchPolopt` class takes care of collecting samples and common diagnostic
information. It also provides an abstraction of the general procedure above, so
that algorithm implementations only need to fill the missing pieces. The core
of the :code:`BatchPolopt` class is the :code:`train()` method:


.. code-block:: py

    def train(self, mdp, policy, baseline, **kwargs):
        # ...
        opt_info = self.init_opt(mdp, policy, baseline)
        for itr in xrange(self.start_itr, self.n_itr):
            samples_data = self.obtain_samples(itr, mdp, policy, baseline)
            opt_info = self.optimize_policy(itr, policy, samples_data, opt_info)
            params = self.get_itr_snapshot(
                itr, mdp, policy, baseline, samples_data, opt_info)
            logger.save_itr_params(itr, params)
            # ...

The :code:`obtain_samples` is implemented. The derived class needs to provide
implementation for :code:`init_opt`, which initializes the computation graph,
:code:`optimize_policy`, which updates the policy based on the collected data,
and :code:`get_itr_snapshot`, which returns a dictionary of objects to be persisted
per iteration.

The :code:`BatchPolopt` class powers quite a few algorithms:

- Vanilla Policy Gradient: :code:`rllab/algo/vpg.py`

- Natural Policy Gradient: :code:`rllab/algo/npg.py`

- Reward-Weighted Regression: :code:`rllab/algo/erwr.py`

- Trust Region Policy Optimization: :code:`rllab/algo/trpo.py`

- Relative Entropy Policy Search: :code:`rllab/algo/reps.py`

To give an illustration, here's how we might implement :code:`init_opt` for VPG
(the actual code in :code:`rllab/algo/vpg.py` is longer due to the need to log
extra diagnostic information):

.. code-block:: py

    from rllab.misc.ext import extract, compile_function, new_tensor

    # ...

    def init_opt(self, mdp, policy, baseline):
        # new_tensor() constructs a tensor of the specified dimension and data
        # type
        # we need this rather than simply constructing a matrix since some MDPs
        # may have observations with multiple dimensions
        obs_var = new_tensor(
            'obs',
            ndim=1+len(mdp.observation_shape),
            dtype=mdp.observation_dtype
        )
        advantage_var = TT.vector('advantage')
        action_var = TT.matrix('action', dtype=mdp.action_dtype)
        log_prob = policy.get_log_prob_sym(obs_var, action_var)
        # formulate as a minimization problem
        # The gradient of the surrogate objective is the policy gradient
        surr_obj = - TT.mean(log_prob * advantage_var)
        updates = self.update_method(
            surr_obj, policy.get_params(trainable=True))
        input_list = [obs_var, advantage_var, action_var]
        # compile_function() is a wrapper around theano.function with some
        # default flags set
        f_update = compile_function(
            inputs=input_list,
            outputs=None,
            updates=updates,
        )
        return dict(
            f_update=f_update,
        )

The code is very similar to what we implemented in the basic version. Note that
at the end of the function, we return a dictionary containing the compiled
function which we can use later.

Here's how we might implement :code:`optimize_policy`:

.. code-block:: py

    def optimize_policy(self, itr, policy, samples_data, opt_info):
        f_update = opt_info["f_update"]
        # extract() takes a dictionary and a list of keys, and returns a tuple
        # of corresponding entries
        inputs = extract(
            samples_data,
            "observations", "advantages", "actions"
        )
        f_update(*inputs)
        return opt_info


Parallel Sampling
=================

The :code:`rllab.parallel_sampler` module takes care of parallelizing the
sampling process and aggregating the collected trajectory data. It is used
by the :code:`BatchPolopt` class like below:

.. code-block:: py

    # At the beginning of training, we need to register the mdp and the policy
    # onto the parallel_sampler
    parallel_sampler.populate_task(mdp, policy)

    # ...

    # Within each iteration, we just need to update the policy parameters to
    # each worker
    cur_params = policy.get_param_values()

    parallel_sampler.request_samples(
        policy_params=cur_params,
        max_samples=self.batch_size,
        max_path_length=self.max_path_length,
        whole_paths=self.whole_paths,
    )

    paths = parallel_sampler.collect_paths()

The returned :code:`paths` is a list of dictionaries with keys :code:`rewards`,
:code:`observations`, :code:`actions`, :code:`pdists`, where :code:`pdists`
contains minimally sufficient information about each action distribution. For
a gaussian distribution with diagonal variance, this would be the means and
standard deviations.

After collecting the trajectories, the :code:`obtain_samples` method in the
:code:`BatchPolopt` class computes the empirical returns and advantages by
using the baseline specified through command line arguments (we'll talk about
this below). Then it trains the baseline using the collected data, and
concatenates all rewards, observations, etc. together to form a single huge
tensor, just as we did for the basic algorithm implementation.

One different semantics from the basic implementation is that, rather than
collecting a fixed number of trajectories with potentially different number
of steps per trajectory (if the MDP implements a termination condition), we
specify a desired total number of samples (i.e. time steps) per iteration. The
number of trajectories collected will be around this number, although sometimes
slightly larger, to make sure that all trajectories are run until either the
horizon or the termination condition is met.


Command-line Arguments
======================

We would like to make the algorithms (and MDPs) as flexible and experimentable
as possible by exposing most of its configurations through command line
arguments. This is accomplished by the :code:`rllab.misc.autoargs` module and
the :code:`scripts/run_scripts.py` file.

Recall that for the basic implementation, we have quite a few hyper-parameters,
like the learning rate and the discount factor. We could expose them as command
line arguments through decorating the :code:`__init__` method of the algorithm
class like below:

.. code-block:: py

    from rllab.misc import autoargs

    # ...

    class VPG(BatchPolopt):

        @autoargs.arg("discount", type=float, help="Discount.")
        @autoargs.arg("learning_rate", type=float, help="Learning rate.")
        def __init__(self, discount=0.99, learning_rate=0.01):
            self.discount = discount
            self.learning_rate = learning_rate
            # ...

Then, we could configure these parameter from the command line like
:code:`--algo_discount 0.98`, etc.

To inherit command line arguments of base classes, you can use the
:code:`autoargs.inherit` decorator like below:

.. code-block:: py

    from rllab.misc import autoargs

    # ...
    class VPG(BatchPolopt):

        @autoargs.inherit(BatchPolopt.__init__)
        def __init__(self, *args, **kwargs):
            # ...
            super(VPG, self).__init__(**kwargs)

Logging
=======
