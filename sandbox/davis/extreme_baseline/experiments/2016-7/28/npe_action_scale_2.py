


info = """In order to see the optimal policy contribute more significantly to empirical return, we
vary the scale of the action's contribution to the reward (the remaining contribution is due to
known random noise)"""

from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.davis.envs.noisy_point_env import NoisyPointEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.extreme_linear_baseline import ExtremeLinearBaseline
from rllab.envs.normalized_env import normalize
from rllab.algos.trpo import TRPO
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

import sys
import argparse

stub(globals())

from rllab.misc.instrument import VariantGenerator

N_ITR = 100
N_ITR_DEBUG = 5


def experiment_variant_generator():
    vg = VariantGenerator()
    vg.add("env_noise", [1])
    vg.add("action_scale", [100])
    vg.add("env",
           lambda env_noise, action_scale: [normalize(NoisyPointEnv(env_noise=env_noise,
                                                                    end_early=False,
                                                                    reward_fn='noise-action',
                                                                    action_scale=action_scale))],
           hide=True)
    vg.add("batch_size", [1000], hide=True)
    vg.add("step_size", [0.01], hide=True)
    vg.add("max_path_length", [10], hide=True)
    vg.add("discount", [0.99], hide=True)
    vg.add("seed", list(range(20)), hide=True)
    vg.add("lookahead", [0, 1, 5, 10])
    vg.add("baseline",
           lambda env, lookahead, max_path_length: [
               LinearFeatureBaseline(env.spec),
               ExtremeLinearBaseline(env.spec,
                                     lookahead=lookahead,
                                     use_env_noise=True,
                                     max_path_length=max_path_length)],
           hide=True)
    return vg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--local",
                        help="run experiment locally",
                        action="store_true")
    parser.add_argument("-v", "--visualize",
                        help="visualize simulations during experiment",
                        action="store_true")
    parser.add_argument("-d", "--debug",
                        help="run in debug mode (only one configuration; don't terminate machines)",
                        action="store_true")
    parser.add_argument("-i", "--info",
                        help="display experiment info (without running experiment)",
                        action="store_true")
    parser.add_argument("-n", "--n-itr", type=int,
                        help="number of iterations (overrides default; mainly for debugging)")
    args = parser.parse_args()
    if args.info:
        print(info.replace('\n', ' '))
        sys.exit(0)
    return args

if __name__ == '__main__':
    args = parse_args()
    exp_prefix = __file__.split('/')[-1][:-3].replace('_', '-')
    N_ITR = args.n_itr if args.n_itr is not None else N_ITR_DEBUG if args.debug else N_ITR
    vg = experiment_variant_generator()
    variants = vg.variants()
    print("Running experiment {}.".format(exp_prefix))
    print("Number of experiments to run: {}.".format(len(variants)))
    for variant in variants:
        exp_name = vg.to_name_suffix(variant)
        if exp_name == '':
            exp_name = None

        policy = GaussianMLPPolicy(
            env_spec=variant["env"].spec,
            hidden_sizes=(32, 32)
        )

        algo = TRPO(
            env=variant["env"],
            policy=policy,
            baseline=variant["baseline"],
            batch_size=variant["batch_size"],
            whole_paths=True,
            max_path_length=variant["max_path_length"],
            n_itr=N_ITR,
            discount=variant["discount"],
            step_size=variant["step_size"],
            plot=args.visualize and args.local,
        )

        run_experiment_lite(
            algo.train(),
            n_parallel=1,
            snapshot_mode="last",
            seed=variant["seed"],
            plot=args.visualize and args.local,
            mode="local" if args.local else "ec2",
            exp_prefix=exp_prefix,
            terminate_machine=not args.debug
        )

        if args.debug:
            break  # Only run first experiment variant in debug mode
