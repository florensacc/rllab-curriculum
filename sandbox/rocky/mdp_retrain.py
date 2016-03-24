import os
os.environ["THEANO_FLAGS"] = "device=cpu"
from rllab.env.mujoco import SwimmerMDP
from rllab.env.normalized_mdp import NormalizedMDP
from rllab.env.mujoco import HopperMDP
from rllab.env.mujoco import Walker2DMDP
from rllab.env.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.env.mujoco import AntMDP
from rllab.env.mujoco import SimpleHumanoidMDP
from rllab.env.mujoco.humanoid_mdp import HumanoidMDP
from rllab.policy.mean_std_nn_policy import MeanStdNNPolicy
from rllab.baseline.linear_feature_baseline import LinearFeatureBaseline
from rllab.algo.trpo import TRPO
from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())

mdp_classes = [
    SwimmerMDP,
    HopperMDP,
    Walker2DMDP,
    HalfCheetahEnv,
    AntMDP,
    SimpleHumanoidMDP,
    HumanoidMDP,
]

for mdp_class in mdp_classes:
    mdp = NormalizedMDP(mdp=mdp_class())
    algo = TRPO(
        batch_size=50000,
        whole_paths=True,
        max_path_length=500,
        n_itr=500,
        discount=0.99,
        step_size=0.01,
    )
    policy = GaussianMLPPolicy(
        mdp=mdp,
        hidden_sizes=(100, 50, 25),
        # nonlinearity='lasagne.nonlinearities.rectified',
    )
    baseline = LinearFeatureBaseline(
        mdp=mdp
    )
    run_experiment_lite(
        algo.train(mdp=mdp, policy=policy, baseline=baseline),
        exp_prefix="debug_test",
        n_parallel=4,
        snapshot_mode="last",
        seed=1,
        mode="ec2",
    )
    break
