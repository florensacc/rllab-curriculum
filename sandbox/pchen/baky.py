import os

from rllab.algo.bake_ppo import BakePPO
from rllab.algo.ppo import PPO
from rllab.policy.bake_mean_std_nn_policy import BakeMeanStdNNPolicy

os.environ["THEANO_FLAGS"] = "device=cpu"
from rllab.mdp.mujoco.swimmer_mdp import SwimmerMDP
from rllab.mdp.normalized_mdp import NormalizedMDP
from rllab.mdp.mujoco.hopper_mdp import HopperMDP
from rllab.mdp.mujoco.walker2d_mdp import Walker2DMDP
from rllab.mdp.mujoco.half_cheetah_mdp import HalfCheetahMDP
from rllab.mdp.mujoco.ant_mdp import AntMDP
from rllab.mdp.mujoco.simple_humanoid_mdp import SimpleHumanoidMDP
from rllab.mdp.mujoco.humanoid_mdp import HumanoidMDP
from rllab.policy.mean_std_nn_policy import MeanStdNNPolicy
from rllab.baseline.linear_feature_baseline import LinearFeatureBaseline
from rllab.algo.trpo import TRPO
from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())

mdp_classes = [
    # SwimmerMDP,
    # HopperMDP,
    Walker2DMDP,
    HalfCheetahMDP,
    # AntMDP,
    # SimpleHumanoidMDP,
    # HumanoidMDP,
]

for ss in [0.005, 0.01]:
    for mdp_class in mdp_classes:
        mdp = NormalizedMDP(mdp=mdp_class())
        algo = PPO(
            batch_size=50000,
            whole_paths=True,
            max_path_length=200,
            n_itr=500,
            discount=0.99,
            step_size=ss,
        )
        bake_algo = BakePPO(
            batch_size=50000,
            whole_paths=True,
            max_path_length=200,
            n_itr=500,
            discount=0.99,
            step_size=ss,
        )
        policy = MeanStdNNPolicy(
            mdp=mdp,
            hidden_sizes=(100, 50, 25),
            # nonlinearity='lasagne.nonlinearities.rectified',
        )
        bake_policy = BakeMeanStdNNPolicy(
            mdp=mdp,
            hidden_sizes=(100, 50, 25),
            # nonlinearity='lasagne.nonlinearities.rectified',
        )
        baseline = LinearFeatureBaseline(
            mdp_spec=mdp
        )
        for seed in [1,4,66,777,678]:
            # run_experiment_lite(
            #     algo.train(mdp=mdp, policy=policy, baseline=baseline),
            #     exp_prefix="ppo_loco",
            #     n_parallel=4,
            #     snapshot_mode="last",
            #     seed=seed,
            #     mode="lab_kube",
            # )
            run_experiment_lite(
                bake_algo.train(mdp=mdp, policy=bake_policy, baseline=baseline),
                exp_prefix="ppo_loco",
                n_parallel=4,
                snapshot_mode="last",
                seed=seed,
                mode="lab_kube",
            )

