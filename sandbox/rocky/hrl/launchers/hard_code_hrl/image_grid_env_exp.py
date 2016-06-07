from __future__ import print_function
from __future__ import absolute_import

from sandbox.rocky.hrl.envs.image_grid_env import ImageGridEnv
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.core.network import ConvNetwork
from rllab.algos.trpo import TRPO
from rllab.misc.instrument import stub, run_experiment_lite
import lasagne.nonlinearities as NL

stub(globals())

# Experiment: test if a larger grid needs more time to train

for seed in [1, 11, 21, 31, 41]:
    envs = [
        # ImageGridEnv(size=16, subgoal_interval=4, action_interval=1),
        # ImageGridEnv(size=16, subgoal_interval=4, action_interval=2),
        # ImageGridEnv(size=16, subgoal_interval=4, action_interval=4),
        ImageGridEnv(size=8, subgoal_interval=2, action_interval=1),
        ImageGridEnv(size=8, subgoal_interval=2, action_interval=2),
        ImageGridEnv(size=4, subgoal_interval=1, action_interval=1),
    ]

    for env in envs:
        policy = CategoricalMLPPolicy(
            prob_network=ConvNetwork(
                input_shape=env.observation_space.shape,
                output_dim=env.action_space.n,
                hidden_sizes=(10,),
                conv_filters=(5, 5),
                conv_filter_sizes=(1, 1),
                conv_strides=(1, 1),
                conv_pads=('full', 'full'),
                hidden_nonlinearity=NL.tanh,
                output_nonlinearity=NL.softmax,
            ),
            env_spec=env.spec,
        )
        baseline = ZeroBaseline(env_spec=env.spec)

        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            max_path_length=100,
            batch_size=1000,
            n_itr=100,
        )

        run_experiment_lite(
            algo.train(),
            n_parallel=1,
            exp_prefix="image_grid_small_flat_exp",
            mode="lab_kube",
            snapshot_mode="last",
            seed=seed,
        )
