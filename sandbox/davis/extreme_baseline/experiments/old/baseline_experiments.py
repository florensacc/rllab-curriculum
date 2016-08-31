from __future__ import print_function
from __future__ import absolute_import

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.extreme_linear_baseline import ExtremeLinearBaseline
from rllab.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv
from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
from rllab.envs.mujoco.walker2d_env import Walker2DEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

import itertools

stub(globals())

# Name
exp_prefix = "davis-extreme-baseline-k1-test"

# Settings
local = True
visualize = False
debug = True
n_itr = 500

# Experiment parameters
envs = [
    CartpoleSwingupEnv(),
    # DoublePendulumEnv(),
    # Walker2DEnv(),
    HalfCheetahEnv(),
    ]
gae_lambdas = [1, 0.995, 0.99, 0.98, 0.97]
discounts = [0.99]
step_sizes = [0.1]
seeds = [1, 11, 21, 31]
batch_sizes = [4000]
lookaheads = [0, 1, 2, 3]

# Handling code for experiment configuration
envs = [normalize(env) for env in envs]
plot = local and visualize
mode = "local" if local else "ec2"
terminate_machine = not debug
if debug:
    envs        = [envs[-1]]
    discounts   = [discounts[-1]]
    step_sizes  = [step_sizes[-1]]
    seeds       = [seeds[-1]]
    batch_sizes = [1000]
    exp_prefix += "-DEBUG"

# Experiments
configurations = list(itertools.product(
    envs,
    gae_lambdas,
    discounts,
    step_sizes,
    batch_sizes,
    lookaheads,
    seeds))
if not local:
    print("Number of EC2 instances to launch: {}".format(len(configurations)))
for env, gae_lambda, discount, step_size, batch_size, lookahead, seed in configurations:

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32, 32)
    )

    baseline = ExtremeLinearBaseline(env_spec=env.spec, lookahead=lookahead)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        whole_paths=True,
        max_path_length=200,
        n_itr=n_itr,
        gae_lambda=gae_lambda,
        discount=discount,
        step_size=step_size,
        plot=plot,
    )

    run_experiment_lite(
        algo.train(),
        n_parallel=1,
        snapshot_mode="last",
        seed=seed,
        plot=plot,
        mode=mode,
        exp_prefix=exp_prefix,
        terminate_machine=terminate_machine
    )
