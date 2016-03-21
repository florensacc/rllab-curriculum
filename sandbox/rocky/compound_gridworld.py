import os
os.environ["THEANO_FLAGS"] = "device=cpu"
from rllab.mdp.compound_action_sequence_mdp import CompoundActionSequenceMDP
from rllab.mdp.grid_world_mdp import GridWorldMDP
from rllab.mdp.box2d.cartpole_mdp import CartpoleMDP
from rllab.policy.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.policy.categorical_gru_policy import CategoricalGRUPolicy
from rllab.policy.mean_std_rnn_policy import MeanStdRNNPolicy
from rllab.baseline.linear_feature_baseline import LinearFeatureBaseline
from rllab.baseline.zero_baseline import ZeroBaseline
from rllab.algo.ppo import PPO
from rllab.algo.trpo import TRPO
from rllab.optimizer.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.misc import ext

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
    ]
)
algo = TRPO(
    batch_size=10000,
    whole_paths=True,
    max_path_length=500,
    n_itr=100,
    discount=0.99,
    step_size=0.01,
    # optimizer=PenaltyLbfgsOptimizer(
    #     max_penalty_itr=5,
    # )
)
# policy = CategoricalGRUPolicy(
#     mdp_spec=mdp.spec,
#     include_action=False,
#     hidden_sizes=(32,),
# )
policy = CategoricalMLPPolicy(
    mdp_spec=mdp.spec,
    hidden_sizes=(32,),
)
baseline = ZeroBaseline(mdp_spec=mdp.spec)#LinearFeatureBaseline(
#     mdp_spec=mdp.spec
# )
run_experiment_lite(
    algo.train(mdp=mdp, policy=policy, baseline=baseline),
    exp_prefix="compound_gridworld_4x4_free",
    n_parallel=1,
    snapshot_mode="last",
    seed=1,
    mode="local",
    # log_tabular_only=True,
    env=ext.merge_dict(os.environ, dict(THEANO_FLAGS="optimizer=None"))
)
