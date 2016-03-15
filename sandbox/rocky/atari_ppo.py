import os
os.environ["THEANO_FLAGS"] = "device=cpu"
from rllab.mdp.openai_atari_mdp import AtariMDP
from rllab.policy.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.baseline.linear_feature_baseline import LinearFeatureBaseline
from rllab.algo.ppo import PPO
from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())

for seed in [2,3,4,5]:

    for hidden_size in [32, 64, 128, 256]:

        mdp = AtariMDP(rom_name="pong", obs_type="ram", frame_skip=4)
        algo = PPO(
            batch_size=10000,
            whole_paths=True,
            max_path_length=4500,
            n_itr=200,
            discount=0.99,
            step_size=0.01,
            max_penalty_itr=5,
        )
        policy = CategoricalMLPPolicy(
            mdp=mdp,
            hidden_sizes=[hidden_size, hidden_size]
        )
        baseline = LinearFeatureBaseline(
            mdp=mdp
        )
        run_experiment_lite(
            algo.train(mdp=mdp, policy=policy, baseline=baseline),
            exp_prefix="ppo_atari",
            n_parallel=4,
            snapshot_mode="last",
            seed=5**seed,
            mode="ec2",
        )
