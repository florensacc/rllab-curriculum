from __future__ import print_function
from __future__ import absolute_import

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.extreme_model_baseline import ExtremeModelBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

import lasagne.nonlinearities as NL
import itertools

stub(globals())

# Name
exp_prefix = "model-baseline-experiment-3"

# Settings
local = True
visualize = False
debug = True
n_itr = 200

# Experiment parameters
envs = [HalfCheetahEnv()]
gae_lambdas = [1]
discounts = [0.99]
step_sizes = [0.1]
seeds = [1, 21, 31, 41]
batch_sizes = [4000]
lookaheads = [0, 1, 2, 5, 10]
network_sizes = [(32, 32), (200, 200)]
max_opt_itrs = [20, 100]
nonlinearities = [NL.rectify, NL.tanh]

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
baseline_configurations = list(itertools.product(network_sizes, max_opt_itrs, nonlinearities))
if debug:
    configurations = [configurations[0]]
    baseline_configurations = [baseline_configurations[0]]
    lookaheads = [0, 1, 2]
    n_itr = 5
if not local:
    print("Number of EC2 instances to launch: {}".format(
        ((len(lookaheads) + 1)*len(baseline_configurations) + 1)*len(configurations)))
for env, gae_lambda, discount, step_size, batch_size, seed in configurations:

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
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

    for network_size, max_opt_itr, nonlinearity in baseline_configurations:
        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(32, 32)
        )

        gaussian_baseline_algo = TRPO(
            env=env,
            policy=policy,
            baseline=GaussianMLPBaseline(env_spec=env.spec,
                                         max_opt_itr=max_opt_itr,
                                         regressor_args={'hidden_sizes': network_size,
                                                         'hidden_nonlinearity': nonlinearity}),
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

        for lookahead in lookaheads:

            policy = GaussianMLPPolicy(
                env_spec=env.spec,
                hidden_sizes=(32, 32)
            )

            baseline = ExtremeModelBaseline(env_spec=env.spec,
                                            lookahead=lookahead,
                                            batch_size=batch_size,
                                            discount=discount,
                                            experience_limit=10,
                                            max_opt_itr=max_opt_itr,
                                            nonlinearity=nonlinearity,
                                            network_size=network_size,
                                            )

            model_baseline_algo = TRPO(
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
                model_baseline_algo.train(),
                n_parallel=1,
                snapshot_mode="last",
                seed=seed,
                plot=plot,
                mode=mode,
                exp_prefix=exp_prefix,
                terminate_machine=terminate_machine
            )
