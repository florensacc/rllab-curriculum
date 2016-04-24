from __future__ import print_function
from __future__ import absolute_import

from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.hrl.envs.seaquest_grid_world_env import SeaquestGridWorldEnv

from rllab.algos.trpo import TRPO
from rllab.spaces.discrete import Discrete
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.regressors.categorical_mlp_regressor import CategoricalMLPRegressor
from rllab.core.network import ConvNetwork
from sandbox.rocky.hrl.batch_hrl import BatchHRL
from sandbox.rocky.hrl.subgoal_policy import SubgoalPolicy
from sandbox.rocky.hrl.subgoal_baseline import SubgoalBaseline
from sandbox.rocky.hrl.mi_evaluator.state_based_mi_evaluator import StateBasedMIEvaluator
from sandbox.rocky.hrl.core.network import ConvMergeNetwork
import lasagne.nonlinearities as NL
import sys

HIERARCHICAL = True

stub(globals())

if HIERARCHICAL:

    seed = 11

    batch_size = 4000

    grid_size = 10

    env = SeaquestGridWorldEnv(
        size=grid_size,
        n_bombs=grid_size / 2,
    )

    for n_subgoals in [5, 10, 15, 20, 25, 30]:
        high_network = ConvNetwork(
            input_shape=env.observation_space.shape,
            output_dim=n_subgoals,
            hidden_sizes=(20,),
            conv_filters=(8, 8),
            conv_filter_sizes=(3, 3),
            conv_strides=(1, 1),
            conv_pads=('full', 'full'),
            hidden_nonlinearity=NL.tanh,
            output_nonlinearity=NL.softmax,
        )

        low_network = ConvMergeNetwork(
            input_shape=env.observation_space.shape,
            extra_input_shape=(n_subgoals,),
            output_dim=env.action_space.flat_dim,
            extra_hidden_sizes=(20,),
            hidden_sizes=(20,),
            conv_filters=(8, 8),
            conv_filter_sizes=(3, 3),
            conv_strides=(1, 1),
            conv_pads=('full', 'full'),
            hidden_nonlinearity=NL.tanh,
            output_nonlinearity=NL.softmax,
        )

        policy = SubgoalPolicy(
            env_spec=env.spec,
            high_policy_cls=CategoricalMLPPolicy,
            high_policy_args=dict(prob_network=high_network),
            low_policy_cls=CategoricalMLPPolicy,
            low_policy_args=dict(prob_network=low_network),
            subgoal_space=Discrete(n_subgoals),
            subgoal_interval=3,
        )

        baseline = SubgoalBaseline(
            env_spec=env.spec,
            high_baseline=LinearFeatureBaseline(env_spec=policy.high_env_spec),
            low_baseline=LinearFeatureBaseline(env_spec=policy.low_env_spec),
        )

        mi_evaluator = StateBasedMIEvaluator(
            env_spec=env.spec,
            policy=policy,
            regressor_cls=CategoricalMLPRegressor,
            regressor_args=dict(use_trust_region=False),
        )

        low_algo = TRPO(
            env=env,
            policy=policy.low_policy,
            baseline=baseline.low_baseline,
            discount=0.99,
            step_size=0.01,
        )

        algo = BatchHRL(
            env=env,
            policy=policy,
            baseline=baseline,
            bonus_evaluator=mi_evaluator,
            batch_size=batch_size,
            mi_coeff=0.01,
            max_path_length=100,
            n_itr=100,
            bonus_gradient=True,
            high_algo=TRPO(
                env=env,
                policy=policy.high_policy,
                baseline=baseline.high_baseline,
                discount=0.99,
                step_size=0.01,
            ),
            low_algo=low_algo,
        )

        run_experiment_lite(
            algo.train(),
            exp_prefix="hrl_seaquest",
            snapshot_mode="last",
            seed=seed,
            n_parallel=1,
        )

        sys.exit(0)

else:
    for seed in [11, 21, 31, 41, 51]:
        for size in [10]:  # , 15]:
            env = SeaquestGridWorldEnv(
                size=size,
                n_bombs=size / 2,
                # agent_position=(0, 0),
                # goal_position=(size-1, size-1),
                # bomb_positions=[
                #     (5, 0), (5, 1), (5, 2), (5, 3), (5, 6), (5, 7), (5, 8), (5, 9),
                # ],
            )

            network = ConvNetwork(
                input_shape=env.observation_space.shape,
                output_dim=env.action_space.n,
                hidden_sizes=(20,),
                conv_filters=(8, 8),
                conv_filter_sizes=(3, 3),
                conv_strides=(1, 1),
                conv_pads=('full', 'full'),
                hidden_nonlinearity=NL.tanh,
            )
            policy = CategoricalMLPPolicy(env_spec=env.spec, prob_network=network)
            # policy = CategoricalCNNPolicy(env_spec=env.spec, hidden_sizes=tuple(),)
            baseline = ZeroBaseline(env_spec=env.spec)  # LinearFeatureBaseline(env_spec=env.spec)

            algo = TRPO(
                env=env,
                policy=policy,
                baseline=baseline,
                batch_size=4000,
                max_path_length=100,
                n_itr=200,
            )

            run_experiment_lite(
                algo.train(),
                exp_prefix="seaquest",
                snapshot_mode="last",
                seed=seed,
            )
            # sys.exit(0)
