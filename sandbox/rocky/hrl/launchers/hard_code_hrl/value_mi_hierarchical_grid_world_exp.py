


from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.hrl.envs.hierarchical_grid_world_env import HierarchicalGridWorldEnv

from rllab.algos.trpo import TRPO
from rllab.spaces.discrete import Discrete
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.regressors.categorical_mlp_regressor import CategoricalMLPRegressor
from sandbox.rocky.hrl.batch_hrl import BatchHRL
from sandbox.rocky.hrl.subgoal_policy import SubgoalPolicy
from sandbox.rocky.hrl.subgoal_baseline import SubgoalBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from sandbox.rocky.hrl.mi_evaluator.state_based_value_mi_evaluator import StateBasedValueMIEvaluator
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer

stub(globals())

env = HierarchicalGridWorldEnv(
    high_grid=[
        "SFFFF",
        "FFFFF",
        "FFFFG",
    ],

    low_grid=[
        "SF",
        "FG",
    ],
)

seed = 11

batch_size = 10000

n_subgoals = 15

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
    high_baseline=ZeroBaseline(policy.high_env_spec),
    low_baseline=ZeroBaseline(policy.low_env_spec),
)

mi_evaluator = StateBasedValueMIEvaluator(
    env=env,
    policy=policy,
    discount=0.99,
    state_regressor_args=dict(use_trust_region=True, optimizer=ConjugateGradientOptimizer()),
    state_goal_regressor_args=dict(use_trust_region=True, optimizer=ConjugateGradientOptimizer()),
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
    reward_coeff=1.,
    mi_coeff=0.,
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
    exp_prefix="hrl_value_mi",
    snapshot_mode="last",
    seed=seed,
    n_parallel=1,
)
