import os

from rllab.core.network import ConvNetwork
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.hopper_env import HopperEnv
from rllab.envs.mujoco.simple_humanoid_env import SimpleHumanoidEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.walker2d_env import Walker2DEnv
from rllab.envs.normalized_env import NormalizedEnv
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.pchen.envs.atari import AtariEnv

os.environ["THEANO_FLAGS"] = "device=cpu"
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.algos.trpo import TRPO
from rllab.misc.instrument import stub, run_experiment_lite
import lasagne.nonlinearities as LN

stub(globals())

roms = [
    "pong",
    "space_invaders",
    "seaquest",
    "breakout",
]

for rom in roms:
    mdp = AtariEnv(
        game=rom,
        obs_type='image',
    )
    network = ConvNetwork(
        input_shape=mdp.spec.observation_space.shape,
        output_dim=mdp.spec.action_space.n,
        hidden_sizes=(20,),
        conv_filters=(16, 16),
        conv_filter_sizes=(4, 4),
        conv_strides=(2, 2),
        conv_pads=(0, 0),
    )
    policy = CategoricalMLPPolicy(
        env_spec=mdp.spec,
        prob_network=network,
    )
    baseline = LinearFeatureBaseline(
        mdp.spec
    )
    algo = TRPO(
        env=mdp,
        policy=policy,
        baseline=baseline,
        batch_size=3000,
        whole_paths=True,
        max_path_length=500,
        n_itr=5000,
        discount=0.99,
        step_size=1e-2,
        store_paths=False,
    )
    for seed in [42, 123, 88]:
        run_experiment_lite(
            algo.train(),
            exp_prefix="ss_atari_trpo_expert_run",
            n_parallel=3,
            snapshot_mode="all",
            seed=seed,
            # mode="local",
            mode="lab_kube",
            resouces=dict(
                requests=dict(
                    cpu=8
                )
            ),
            node_selector={
                "aws/type": "g2.2xlarge"
            }
        )
        # import ipdb; ipdb.set_trace()

