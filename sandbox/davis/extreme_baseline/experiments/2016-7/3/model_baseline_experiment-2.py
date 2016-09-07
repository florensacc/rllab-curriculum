


from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.extreme_model_baseline import ExtremeModelBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

import itertools

stub(globals())

# Name
exp_prefix = "model-baseline-experiment-2"

# Settings
local = True
visualize = True
debug = False
n_itr = 200

# Experiment parameters
envs = [HalfCheetahEnv()]
gae_lambdas = [1]
discounts = [0.99]
step_sizes = [0.1]
seeds = [1, 21, 31, 41]
batch_sizes = [4000]
lookaheads = [0, 1, 2, 3, 4, 5]

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
    seeds))
if debug:
    configurations = [configurations[0]]
    lookaheads = [2]
if not local:
    print("Number of EC2 instances to launch: {}".format(len(configurations) * (len(lookaheads) + 2)))
for env, gae_lambda, discount, step_size, batch_size, seed in configurations:

    for lookahead in lookaheads:

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            # The neural network policy should have two hidden layers, each with 32 hidden units.
            hidden_sizes=(32, 32)
        )

        baseline = ExtremeModelBaseline(env_spec=env.spec,
                                        lookahead=lookahead,
                                        batch_size=batch_size,
                                        discount=discount,
                                        experience_limit=10,
                                        )

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

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32, 32)
    )

    linear_baseline_algo = TRPO(
        env=env,
        policy=policy,
        baseline=LinearFeatureBaseline(env_spec=env.spec),
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
        linear_baseline_algo.train(),
        n_parallel=1,
        snapshot_mode="last",
        seed=seed,
        plot=plot,
        mode=mode,
        exp_prefix=exp_prefix,
        terminate_machine=terminate_machine
    )

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32, 32)
    )

    gaussian_baseline_algo = TRPO(
        env=env,
        policy=policy,
        baseline=GaussianMLPBaseline(env_spec=env.spec),
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
        gaussian_baseline_algo.train(),
        n_parallel=1,
        snapshot_mode="last",
        seed=seed,
        plot=plot,
        mode=mode,
        exp_prefix=exp_prefix,
        terminate_machine=terminate_machine
    )
