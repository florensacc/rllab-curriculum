from __future__ import print_function
from __future__ import absolute_import

from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.hrl.hierarchical_grid_world_env import HierarchicalGridWorldEnv

from rllab.algos.trpo import TRPO
from rllab.algos.ppo import PPO
from rllab.spaces.discrete import Discrete
from rllab.spaces.box import Box
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.regressors.categorical_mlp_regressor import CategoricalMLPRegressor
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.hrl.batch_hrl import BatchHRL
import lasagne.nonlinearities as NL
from sandbox.rocky.hrl.subgoal_policy import SubgoalPolicy
from sandbox.rocky.hrl.subgoal_baseline import SubgoalBaseline
from sandbox.rocky.hrl.trpo_bonus import TRPOBonus
from sandbox.rocky.hrl.mi_evaluator.state_based_mi_evaluator import StateBasedMIEvaluator
from sandbox.rocky.hrl.policies.state_goal_gaussian_mlp_policy import StateGoalCategoricalMLPPolicy
from sandbox.rocky.hrl.regressors.state_goal_categorical_mlp_regressor import StateGoalCategoricalMLPRegressor
from sandbox.rocky.hrl.mi_evaluator.goal_based_mi_evaluator import GoalBasedMIEvaluator
from sandbox.rocky.hrl.mi_evaluator.exact_state_based_mi_evaluator import ExactStateBasedMIEvaluator
from sandbox.rocky.hrl.mi_evaluator.continuous_exact_state_based_mi_evaluator import \
    ContinuousExactStateBasedMIEvaluator
import sys

stub(globals())

env = HierarchicalGridWorldEnv(
    high_grid=[
        "SFFFF",
        "FFFFF",
        "FFFFF",
        "FFFFF",
        "FFFFF",
    ],

    low_grid=[
        "SF",
        "FF",
    ],
)

CONTINUOUS = True#False#True#False#True

seed = 11

batch_size = 10000#1000#10000#4000  # 0

subgoal_dim = 1#5#1

component_idx = None

subgoal_space = Box(low=-1, high=1, shape=(subgoal_dim,)) if CONTINUOUS else Discrete(20)

policy = SubgoalPolicy(
    env_spec=env.spec,
    high_policy_cls=GaussianMLPPolicy if CONTINUOUS else CategoricalMLPPolicy,
    high_policy_args=dict(hidden_sizes=(32, 32) if CONTINUOUS else tuple()),
    # low_policy_cls=StateGoalCategoricalMLPPolicy,
    low_policy_cls=CategoricalMLPPolicy,
    low_policy_args=dict(
        hidden_sizes=(32, 32) if CONTINUOUS else tuple(),
        # subgoal_space=subgoal_space,
        # state_hidden_sizes=tuple(),
        # goal_hidden_sizes=(32,),
        # joint_hidden_sizes=(32,)
    ),
    subgoal_space=subgoal_space,
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
    component_idx=component_idx,
)
cont_exact_evaluator = ContinuousExactStateBasedMIEvaluator(
    env=env,
    policy=policy,
    component_idx=component_idx,
)

# mi_evaluator = exact_evaluator

mi_evaluator = StateBasedMIEvaluator(
    env_spec=env.spec,
    policy=policy,
    regressor_cls=CategoricalMLPRegressor,
    regressor_args=dict(
        # state_dim=env.observation_space.flat_dim,
        # goal_dim=subgoal_space.flat_dim,
        # state_hidden_sizes=tuple(),
        # goal_hidden_sizes=(32,),
        # joint_hidden_sizes=(32,),
        # hidden_sizes=(32, 32) if CONTINUOUS else tuple(),
        use_trust_region=False,
        # hidden_nonlinearity=NL.tanh
    ),
    component_idx=component_idx,
    n_subgoal_samples=10,
    use_state_regressor=False,
    # state_regressor_cls=CategoricalMLPRegressor,
    # state_regressor_args=dict(use_trust_region=False, hidden_nonlinearity=NL.tanh),
    logger_delegates=[cont_exact_evaluator] if CONTINUOUS else [cont_exact_evaluator, exact_evaluator],
)

low_algo = TRPO(
    env=policy.low_env_spec,
    policy=policy.low_policy,
    baseline=baseline.low_baseline,
    discount=0.99,
    step_size=0.01,
    optimizer=ConjugateGradientOptimizer(subsample_factor=1.),
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
        env=policy.high_env_spec,
        policy=policy.high_policy,
        baseline=baseline.high_baseline,
        discount=0.99,
        step_size=0.01,
    ),
    low_algo=low_algo,
)

run_experiment_lite(
    algo.train(),
    exp_prefix="hrl_continuous_goals",
    snapshot_mode="last",
    seed=seed,
    n_parallel=1,
)
