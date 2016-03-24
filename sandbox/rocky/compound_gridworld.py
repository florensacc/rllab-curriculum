import os
os.environ["THEANO_FLAGS"] = "device=cpu"
from rllab.envs.compound_action_sequence_mdp import CompoundActionSequenceMDP
from rllab.envs.grid_world_mdp import GridWorldMDP
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.algos.trpo import TRPO
from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())

mdp = CompoundActionSequenceMDP(
    mdp=GridWorldMDP(
        desc=[
            "SFFF",
            "FFFF",
            "FFFF",
            "FFFG"
        ]
    ),
    action_map=[
        [0, 1, 2],
        [0, 0, 0],
        [2, 1, 0],
        [3, 3, 3],
        # [0],
        # [1],
        # [2],
        # [3],
    ],
    obs_include_actions=True
)
algo = TRPO(
    batch_size=5000,
    whole_paths=True,
    max_path_length=500,
    n_itr=100,
    discount=0.99,
    step_size=0.01,
    # optimizer=ConjugateGradientOptimizer(),
    # optimizer=PenaltyLbfgsOptimizer(
    #     max_penalty_itr=5,
    # )
)
# policy = CategoricalGRUPolicy(
#     mdp_spec=mdp.spec,
#     state_include_action=False,
#     hidden_sizes=(32,),
# )
policy = CategoricalMLPPolicy(
    env_spec=mdp.spec,
    hidden_sizes=(32,),
)
baseline = ZeroBaseline(env_spec=mdp.spec)#LinearFeatureBaseline(
#     mdp_spec=mdp.spec
# )
run_experiment_lite(
    algo.train(mdp=mdp, policy=policy, baseline=baseline),
    exp_prefix="compound_gridworld_4x4_free",
    n_parallel=1,
    snapshot_mode="last",
    seed=1,
    mode="local",
    log_tabular_only=True,
    # env=ext.merge_dict(os.environ, dict(THEANO_FLAGS="optimizer=None"))
)
