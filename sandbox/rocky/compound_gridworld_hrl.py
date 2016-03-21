import os
os.environ["THEANO_FLAGS"] = "device=cpu"
from rllab.mdp.compound_action_sequence_mdp import CompoundActionSequenceMDP
from rllab.mdp.grid_world_mdp import GridWorldMDP
from rllab.policy.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.algo.batch_hrl import BatchHRL
from rllab.mdp.subgoal_mdp import SubgoalMDP
from rllab.policy.subgoal_policy import SubgoalPolicy
from rllab.baseline.subgoal_baseline import SubgoalBaseline
from rllab.mi_evaluator.state_given_goal_mi_evaluator import StateGivenGoalMIEvaluator
# from rllab.optimizer.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer

from rllab.baseline.linear_feature_baseline import LinearFeatureBaseline
# from rllab.baseline.zero_baseline import ZeroBaseline
from rllab.algo.ppo import PPO
from rllab.algo.trpo import TRPO
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.misc import ext

stub(globals())

mdp = SubgoalMDP(
    mdp=CompoundActionSequenceMDP(
        mdp=GridWorldMDP(
            desc=[
                "SFFF",
                "FFFF",
                "FFFF",
                "FFFF"
            ]
        ),
        reset_history=True,
        action_map=[
            [0, 1, 2],
            [0, 0, 0],
            [2, 1, 0],
            [3, 3, 3],
            # [0],
            # [1],
            # [2],
            # [3],
        ]
    ),
    n_subgoals=4
)

# mdp = SubgoalMDP(
#     mdp=mdp,
#     n_subgoals=4,
# )

# algo = TRPO(
#     batch_size=10000,
#     whole_paths=True,
#     max_path_length=60,
#     n_itr=100,
#     discount=0.99,
#     step_size=0.01,
#     # optimizer=PenaltyLbfgsOptimizer(
#     #     max_penalty_itr=5,
#     # )
# )
algo = BatchHRL(
    batch_size=10000,
    whole_paths=True,
    max_path_length=60,
    n_itr=100,
    subgoal_interval=3,
    high_algo=TRPO(
        discount=0.99,
        step_size=0.01,
        # optimizer=PenaltyL
    ),
    low_algo=TRPO(
        discount=0.99,
        step_size=0.01,
    ),
)

# policy = CategoricalGRUPolicy(
#     mdp_spec=mdp.spec,
#     include_action=False,
#     hidden_sizes=(32,),
# )

policy = SubgoalPolicy(
    mdp_spec=mdp.spec,
    high_policy=CategoricalMLPPolicy(
        mdp_spec=mdp.high_mdp_spec,
    ),
    low_policy=CategoricalMLPPolicy(
        mdp_spec=mdp.low_mdp_spec,
    ),
)

baseline = SubgoalBaseline(
    mdp_spec=mdp.spec,
    high_baseline=LinearFeatureBaseline(mdp_spec=mdp.high_mdp_spec),
    low_baseline=LinearFeatureBaseline(mdp_spec=mdp.low_mdp_spec),
)

# policy = CategoricalMLPPolicy(
#     mdp_spec=mdp.spec,
#     hidden_sizes=(32,),
# )

# baseline = ZeroBaseline(mdp_spec=mdp.spec)#LinearFeatureBaseline(
#     mdp_spec=mdp.spec
# )

evaluator = StateGivenGoalMIEvaluator(mdp_spec=mdp.spec)
run_experiment_lite(
    algo.train(mdp=mdp, policy=policy, baseline=baseline, bonus_evaluator=evaluator),
    exp_prefix="compound_gridworld_4x4_free_hrl",
    n_parallel=1,
    snapshot_mode="last",
    seed=1,
    mode="local",
    # log_tabular_only=True,
    env=ext.merge_dict(os.environ, dict(THEANO_FLAGS="optimizer=None"))
)
