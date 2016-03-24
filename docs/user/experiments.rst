.. _experiments:


===================
Running Experiments
===================

Configuration using Python
==========================


We provide a simple way for running experiments with user-specified
hyper-parameter settings. An example is given in the
:code:`examples/trpo_cartpole.py` file. You can run it via:

.. code-block:: bash

    python examples/trpo_cartpole.py

Running the script for the first time might take a while for initializing
Theano and compiling the computation graph, which can take a few minutes.
Subsequent runs will be much faster since the compilation is cached. You should
see some log messages like the following:

.. code-block:: text

    using seed 1
    instantiating rllab.mdp.box2d.cartpole_mdp.CartpoleMDP
    instantiating rllab.policy.mean_std_nn_policy.MeanStdNNPolicy
    using argument hidden_sizes with value [32, 32]
    instantiating rllab.baseline.linear_feature_baseline.LinearFeatureBaseline
    instantiating rllab.algo.trpo.TRPO
    using argument batch_size with value 4000
    using argument whole_paths with value True
    using argument n_itr with value 40
    using argument step_size with value 0.01
    using argument discount with value 0.99
    using argument max_path_length with value 100
    using seed 0
    0%                          100%
    [##############################] | ETA: 00:00:00
    Total time elapsed: 00:00:02
    2016-02-14 14:30:56.631891 PST | [trpo_cartpole] itr #0 | fitting baseline...
    2016-02-14 14:30:56.677086 PST | [trpo_cartpole] itr #0 | fitted
    2016-02-14 14:30:56.682712 PST | [trpo_cartpole] itr #0 | optimizing policy
    2016-02-14 14:30:56.686587 PST | [trpo_cartpole] itr #0 | computing loss before
    2016-02-14 14:30:56.698566 PST | [trpo_cartpole] itr #0 | performing update
    2016-02-14 14:30:56.698676 PST | [trpo_cartpole] itr #0 | computing descent direction
    2016-02-14 14:31:26.241657 PST | [trpo_cartpole] itr #0 | descent direction computed
    2016-02-14 14:31:26.241828 PST | [trpo_cartpole] itr #0 | performing backtracking
    2016-02-14 14:31:29.906126 PST | [trpo_cartpole] itr #0 | backtracking finished
    2016-02-14 14:31:29.906335 PST | [trpo_cartpole] itr #0 | computing loss after
    2016-02-14 14:31:29.912287 PST | [trpo_cartpole] itr #0 | optimization finished
    2016-02-14 14:31:29.912483 PST | [trpo_cartpole] itr #0 | saving snapshot...
    2016-02-14 14:31:29.914311 PST | [trpo_cartpole] itr #0 | saved
    2016-02-14 14:31:29.915302 PST | -----------------------  -------------
    2016-02-14 14:31:29.915365 PST | Iteration                   0
    2016-02-14 14:31:29.915410 PST | Entropy                     1.41894
    2016-02-14 14:31:29.915452 PST | Perplexity                  4.13273
    2016-02-14 14:31:29.915492 PST | AverageReturn              68.3242
    2016-02-14 14:31:29.915533 PST | StdReturn                  42.6061
    2016-02-14 14:31:29.915573 PST | MaxReturn                 369.864
    2016-02-14 14:31:29.915612 PST | MinReturn                  19.9874
    2016-02-14 14:31:29.915651 PST | AverageDiscountedReturn    65.5314
    2016-02-14 14:31:29.915691 PST | NumTrajs                 1278
    2016-02-14 14:31:29.915730 PST | ExplainedVariance           0
    2016-02-14 14:31:29.915768 PST | AveragePolicyStd            1
    2016-02-14 14:31:29.921911 PST | BacktrackItr                2
    2016-02-14 14:31:29.922008 PST | MeanKL                      0.00305741
    2016-02-14 14:31:29.922054 PST | MaxKL                       0.0360272
    2016-02-14 14:31:29.922096 PST | LossBefore                 -0.0292939
    2016-02-14 14:31:29.922146 PST | LossAfter                  -0.0510883
    2016-02-14 14:31:29.922186 PST | -----------------------  -------------


You can open the example file to understand what it's doing, which is
self-documented via comments.


Configuration using Command Line
================================


The example file above actually constructs a bash command, which calls the
script :code:`scripts/run_experiment.py`. Alternatively, you can call this
script directly. The command that corresponds to the experiment above is:

.. code-block:: bash

    python scripts/run_experiment.py \
        --mdp box2d.cartpole_mdp
        --normalize_mdp True \
        --policy mean_std_nn_policy \
        --policy_hidden_sizes 32 32 \
        --baseline linear_feature_baseline \
        --exp_name trpo_cartpole \
        --algo trpo \
        --algo_discount 0.99 \
        --algo_whole_paths True \
        --algo_step_size 0.01 \
        --algo_n_itr 40 \
        --algo_max_path_length 100 \
        --algo_batch_size 4000 \
        --n_parallel 1 \
        --snapshot_mode last \
        --seed 1

You can see a list of supported configuration parameters by running:

.. code-block:: bash

    python scripts/run_experiment.py --help

Each of the specific MDP and algorithm might have additional configuration
parameters. You can view further help on these by, e.g. running

.. code-block:: bash

    python scripts/run_experiment.py --algo trpo --more_help
