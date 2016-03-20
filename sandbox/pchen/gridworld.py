import os
os.environ["THEANO_FLAGS"] = "device=cpu"
from rllab.mdp.compound_action_sequence_mdp import CompoundActionSequenceMDP
from rllab.mdp.grid_world_mdp import GridWorldMDP
from rllab.policy.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.baseline.linear_feature_baseline import LinearFeatureBaseline
from rllab.algo.ppo import PPO
from rllab.optimizer.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())

mdp=GridWorldMDP(
        desc=[
            "SFFF",
            "FFFF",
            "FFFF",
            "FFFG"
        ]
    )

algo = PPO(
    batch_size=1000,
    whole_paths=True,
    max_path_length=100,
    n_itr=100,
    discount=0.99,
    step_size=0.01,
    optimizer=PenaltyLbfgsOptimizer(
        max_penalty_itr=5,
    )
)
policy = CategoricalMLPPolicy(
    mdp_spec=mdp.spec,
    hidden_sizes=(32,),
)
baseline = LinearFeatureBaseline(
    mdp_spec=mdp.spec
)
run_experiment_lite(
    algo.train(mdp=mdp, policy=policy, baseline=baseline),
    exp_prefix="gridworld",
    n_parallel=1,
    snapshot_mode="last",
    seed=1,
    mode="lab_kube",
    dry=False,
)
