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

envs = [
    # ImageGridEnv(size=20, subgoal_interval=4, action_interval=1),
    # ImageGridEnv(size=20, subgoal_interval=4, action_interval=4),
    ImageGridEnv(size=16, subgoal_interval=4, action_interval=1),
    ImageGridEnv(size=16, subgoal_interval=4, action_interval=4),
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
        max_path_length=200,
        batch_size=5000,
        n_iter=100,
    )

    run_experiment_lite(
        algo.train(),
        n_parallel=1,
        exp_prefix="image_grid_exp",
        snapshot_mode="last"
    )
