from __future__ import absolute_import
from __future__ import print_function

from sandbox.rocky.hrl.batch_hrl import BatchHRL
from sandbox.rocky.hrl.hierarchical_grid_world_env import HierarchicalGridWorldEnv
from sandbox.rocky.hrl.subgoal_baseline import SubgoalBaseline
from sandbox.rocky.hrl.subgoal_policy import SubgoalPolicy

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.regressors.categorical_mlp_regressor import CategoricalMLPRegressor
from rllab.spaces.discrete import Discrete
from sandbox.rocky.hrl.mi_evaluator.exact_state_based_mi_evaluator import ExactStateBasedMIEvaluator
from sandbox.rocky.hrl.mi_evaluator.goal_based_mi_evaluator import GoalBasedMIEvaluator

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

seed = 11

batch_size = 200000

n_subgoals = 5

for use_entropy in [True, False]:


    #for n_subgoals in [5, 10, 15, 20, 25, 30]:#= 5

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

    mi_evaluator = GoalBasedMIEvaluator(
        env_spec=env.spec,
        policy=policy,
        regressor_cls=CategoricalMLPRegressor,
        regressor_args=dict(use_trust_region=False),
        component_idx=0,
        use_entropy=use_entropy,
        logger_delegate=exact_evaluator,
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
        exp_prefix="hrl_goal_based_more_goals_ent",
        snapshot_mode="last",
        seed=seed,
        n_parallel=4,
    )
