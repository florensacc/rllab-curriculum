from __future__ import print_function
from __future__ import absolute_import
import os
os.environ['THEANO_FLAGS'] = 'device=cpu'

from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.hrl.envs.seaquest_grid_world_env import SeaquestGridWorldEnv

from rllab.algos.trpo import TRPO
from rllab.algos.ppo import PPO
from rllab.spaces.discrete import Discrete
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.regressors.categorical_mlp_regressor import CategoricalMLPRegressor
from rllab.core.network import ConvNetwork
from sandbox.rocky.hrl.batch_hrl import BatchHRL
from sandbox.rocky.hrl.subgoal_policy import SubgoalPolicy
from sandbox.rocky.hrl.subgoal_baseline import SubgoalBaseline
from sandbox.rocky.hrl.trpo_bonus import TRPOBonus
from sandbox.rocky.hrl.mi_evaluator.component_state_based_mi_evaluator import ComponentStateBasedMIEvaluator
from sandbox.rocky.hrl.mi_evaluator.goal_based_mi_evaluator import GoalBasedMIEvaluator
from sandbox.rocky.hrl.mi_evaluator.exact_state_based_mi_evaluator import ExactStateBasedMIEvaluator
import lasagne.nonlinearities as NL
import sys

HIERARCHICAL = False

stub(globals())

if HIERARCHICAL:

    seed = 11

    batch_size = 10000#200000

    for n_subgoals in [5, 10, 15, 20, 25, 30]:

        policy = SubgoalPolicy(
            env_spec=env.spec,
            high_policy_cls=CategoricalMLPPolicy,
            high_policy_args=dict(hidden_sizes=tuple()),
            low_policy_cls=CategoricalMLPPolicy,
            low_policy_args=dict(hidden_sizes=tuple()),
            subgoal_space=Discrete(n_subgoals),
            subgoal_interval=3,
        )

        baseline = SubgoalBaseline(
            env_spec=env.spec,
            high_baseline=LinearFeatureBaseline(env_spec=policy.high_env_spec),
            low_baseline=LinearFeatureBaseline(env_spec=policy.low_env_spec),
        )

        exact_evaluator = ExactStateBasedMIEvaluator(
            env=env,
            policy=policy,
            component_idx=0,
        )

        # mi_evaluator = exact_evaluator

        mi_evaluator = ComponentStateBasedMIEvaluator(
            env_spec=env.spec,
            policy=policy,
            regressor_cls=CategoricalMLPRegressor,
            regressor_args=dict(use_trust_region=False),
            component_idx=0,
            logger_delegate=exact_evaluator
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
            exp_prefix="hrl_more_goals",
            snapshot_mode="last",
            seed=seed,
            n_parallel=1,
        )

    # sys.exit(0)

else:
    for size in [5, 7, 10, 15]:
        env = SeaquestGridWorldEnv(size=size)

        network = ConvNetwork(
            input_shape=env.observation_space.shape,
            output_dim=env.action_space.n,
            hidden_sizes=(20,),
            conv_filters=(16, 16),
            conv_filter_sizes=(3, 3),
            conv_strides=(1, 1),
            conv_pads=('full', 'full'),
            hidden_nonlinearity=NL.tanh,
        )
        policy = CategoricalMLPPolicy(env_spec=env.spec, prob_network=network)
        # policy = CategoricalCNNPolicy(env_spec=env.spec, hidden_sizes=tuple(),)
        baseline = ZeroBaseline(env_spec=env.spec)#LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=4000,
            max_path_length=100,
            n_itr=100,
        )

        run_experiment_lite(
            algo.train(),
            exp_prefix="seaquest",
            snapshot_mode="last",
            seed=11,
            env=dict(THEANO_FLAGS='device=cpu'),
        )
        # sys.exit(0)
