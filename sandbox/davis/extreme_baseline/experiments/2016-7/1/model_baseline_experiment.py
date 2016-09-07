


from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.extreme_linear_baseline import ExtremeLinearBaseline
from rllab.baselines.extreme_model_baseline import ExtremeModelBaseline
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
exp_prefix = "davis-model-baseline-experiment"

# Settings
local = True
visualize = False
debug = True
n_itr = 20

# Experiment parameters
envs = [
    # CartpoleSwingupEnv(),
    # DoublePendulumEnv(),
    # Walker2DEnv(),
    HalfCheetahEnv(),
    ]
gae_lambdas = [1]
discounts = [0.99]
step_sizes = [0.1]
seeds = [1]
batch_sizes = [4000]
lookaheads = [1]

# Handling code for experiment configuration
envs = [normalize(env) for env in envs]
plot = local and visualize
mode = "local" if local else "ec2"
terminate_machine = not debug
if debug:
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
if debug:
    configurations = [configurations[0]]
if not local:
    print("Number of EC2 instances to launch: {}".format(len(configurations)))
for env, gae_lambda, discount, step_size, batch_size, lookahead, seed in configurations:

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32, 32)
    )

    baseline = ExtremeModelBaseline(env_spec=env.spec, lookahead=lookahead, discount=discount)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        whole_paths=True,
        max_path_length=100,
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
