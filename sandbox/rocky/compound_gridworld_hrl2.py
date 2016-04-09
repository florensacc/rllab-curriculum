import os

os.environ["THEANO_FLAGS"] = "device=cpu"
from rllab.envs.compound_action_sequence_env import CompoundActionSequenceEnv
from rllab.envs.grid_world_env import GridWorldEnv
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.algos.batch_hrl2 import BatchHRL2
from rllab.algos.batch_hrl3 import BatchHRL3
from rllab.algos.batch_hrl import BatchHRL
from rllab.spaces import Discrete
from rllab.policies.subgoal_policy import SubgoalPolicy
from rllab.regressors.categorical_mlp_regressor import CategoricalMLPRegressor
from rllab.baselines.subgoal_baseline import SubgoalBaseline
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.mi_evaluator.state_given_goal_mi_evaluator import StateGivenGoalMIEvaluator
from rllab.mi_evaluator.exact_state_given_goal_mi_evaluator2 import ExactStateGivenGoalMIEvaluator2
from rllab.mi_evaluator.zero_mi_evaluator import ZeroMIEvaluator
from rllab.mi_evaluator.exact_state_given_goal_mi_evaluator import ExactStateGivenGoalMIEvaluator
from rllab.algos.trpo import TRPO
from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())

env = GridWorldEnv(
    desc=[
        "SFFF",
        "FFFF",
        "FFFF",
        "FFFG"
    ]
)

level = "hrl_level2"

level_configs = dict(
    hrl_level0=dict(
        subgoal_interval=1,
        action_map=[
            [0],
            [1],
            [2],
            [3],
        ],
    ),
    hrl_level1=dict(
        subgoal_interval=3,
        action_map=[
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
        ],
    ),
    hrl_level2=dict(
        subgoal_interval=3,
        action_map=[
            [0, 1, 2],
            [0, 0, 0],
            [2, 1, 0],
            [3, 3, 3],
        ],
    ),
)

env = CompoundActionSequenceEnv(
    wrapped_env=env,
    reset_history=True,
    # obs_include_actions=True,
    action_map=level_configs[level]["action_map"],
)

policy = SubgoalPolicy(
    env_spec=env.spec,
    subgoal_interval=level_configs[level]["subgoal_interval"],
    subgoal_space=Discrete(n=4),
    high_policy_cls=CategoricalMLPPolicy,
    high_policy_args=dict(),
    low_policy_cls=CategoricalMLPPolicy,
    low_policy_args=dict()
)

baseline = SubgoalBaseline(
    env_spec=env.spec,
    high_baseline=LinearFeatureBaseline(
        env_spec=policy.high_env_spec,
    ),
    low_baseline=LinearFeatureBaseline(
        env_spec=policy.low_env_spec,
    ),
    # high_baseline=GaussianMLPBaseline(
    #     env_spec=env.spec.high_env_spec,
    #     regressor_args=dict(
    #         learn_std=False,
    #         optimizer=ConjugateGradientOptimizer(),
    #     ),
    # ),
    # low_baseline=GaussianMLPBaseline(
    #     env_spec=env.spec.low_env_spec,
    #     regressor_args=dict(
    #         learn_std=False,
    #         optimizer=ConjugateGradientOptimizer(),
    #     ),
    # ),
)

exact_evaluator = ExactStateGivenGoalMIEvaluator(
    env=env,
    policy=policy,
)
exact_evaluator2 = ExactStateGivenGoalMIEvaluator2(
    env=env,
    policy=policy,
)
approx_evaluator = StateGivenGoalMIEvaluator(
    env_spec=env.spec,
    policy=policy,
    # high_policy_distribution=high_policy.distribution,
    # low_policy_distribution=low_policy.distribution,
    regressor_cls=CategoricalMLPRegressor,
    regressor_args=dict(
        optimizer=ConjugateGradientOptimizer(),
    ),
    logger_delegate=exact_evaluator,
)


evaluators = [ZeroMIEvaluator(env_spec=env.spec, policy=policy)]#approx_evaluator, exact_evaluator]

for evaluator in evaluators:
    algo = BatchHRL(
        env=env,
        policy=policy,
        baseline=baseline,
        bonus_evaluator=evaluator,
        # addn_evaluator=exact_evaluator2,
        batch_size=10000,
        whole_paths=True,
        max_path_length=60,
        n_itr=100,
        high_algo=TRPO(
            env=env,
            policy=policy.high_policy,
            baseline=baseline.high_baseline,
            discount=0.99,
            step_size=0.01,
        ),
        low_algo=TRPO(
            env=env,
            policy=policy.low_policy,
            baseline=baseline.low_baseline,
            discount=0.99,
            step_size=0.01,
        ),
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix=level + "_approx_exact_cmp",
        n_parallel=4,
        snapshot_mode="last",
        seed=111,
        mode="local",
        # env=ext.merge_dict(os.environ, dict(THEANO_FLAGS="optimizer=None"))
    )
