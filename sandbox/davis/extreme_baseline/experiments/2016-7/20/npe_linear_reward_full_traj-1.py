from __future__ import print_function
from __future__ import absolute_import

from rllab.algos.trpo import TRPO
from sandbox.davis.envs.noisy_point_env import NoisyPointEnv
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import normalize
from rllab.baselines.extreme_linear_baseline import ExtremeLinearBaseline

import itertools

stub(globals())

# Name
exp_prefix = "npe-linear-full-traj-1"

# Settings
local = False
debug = False
visualize = False
n_itr = 100

# Experiment parameters
seeds = [1, 21, 31, 41, 51]
batch_sizes = [1000, 4000]
env_noises = [0, 0.1, 0.2, 0.5, 1]
lookaheads = [0, 1, 2, 5, 10]
max_path_lengths = [10]

# Handling code for experiment configuration
plot = local and visualize
mode = "local" if local else "ec2"
terminate_machine = not debug
if debug:
    exp_prefix += "-DEBUG"

# Experiments
configurations = list(itertools.product(
    batch_sizes,
    seeds,
    env_noises,
    lookaheads,
    max_path_lengths,
    ))
if debug:
    configurations = [configurations[-1]]
    n_itr = 2
if not local:
    print("Number of EC2 instances to launch: {}".format(len(configurations)))
for batch_size, seed, env_noise, lookahead, mpl in configurations:
    env = normalize(NoisyPointEnv(env_noise, end_early=False, reward_fn='noise'))

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(32, 32)
    )

    baseline = ExtremeLinearBaseline(
        env.spec,
        lookahead=lookahead,
        use_env_noise=True,
        max_path_length=mpl,
        validate=True,
        )

    model_baseline_algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        whole_paths=True,
        max_path_length=mpl,
        n_itr=n_itr,
        discount=0.99,
        step_size=0.01,
        plot=plot,
    )

    run_experiment_lite(
        model_baseline_algo.train(),
        n_parallel=1,
        snapshot_mode="last",
        seed=seed,
        plot=plot,
        mode=mode,
        exp_prefix=exp_prefix,
        terminate_machine=terminate_machine
    )
