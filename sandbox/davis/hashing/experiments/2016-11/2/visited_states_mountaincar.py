from __future__ import print_function
from __future__ import absolute_import

info = """Trying different dim_keys to encourage more hashing collisions."""

from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.davis.hashing.algos.bonus_trpo import BonusTRPO
from sandbox.davis.hashing.bonus_evaluators.hashing_bonus_evaluator import HashingBonusEvaluator
from sandbox.davis.hashing.envs.mountain_car_env_x import MountainCarEnvX
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.normalized_env import normalize

import sys
import argparse

stub(globals())

from rllab.misc.instrument import VariantGenerator

N_ITR = 15
N_ITR_DEBUG = 5

envs = [(MountainCarEnvX())]


def experiment_variant_generator():
    vg = VariantGenerator()
    vg.add("env", map(TfEnv, envs), hide=True)
    vg.add("batch_size", [5000], hide=True)
    vg.add("step_size", [0.01], hide=True)
    vg.add("max_path_length", [500], hide=True)
    vg.add("discount", [0.99], hide=True)
    vg.add("seed", [2], hide=True)
    vg.add("bonus_coeff", [0.1, 0])
    vg.add("dim_key", [128])
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
    if args.debug:
        exp_prefix += '-DEBUG'
    N_ITR = args.n_itr if args.n_itr is not None else N_ITR_DEBUG if args.debug else N_ITR
    vg = experiment_variant_generator()
    variants = vg.variants()
    print("Running experiment {}.".format(exp_prefix))
    print("Number of experiments to run: {}.".format(len(variants) if not args.debug else 1))
    for variant in variants:
        exp_name = vg.to_name_suffix(variant)
        if exp_name == '':
            exp_name = None

        policy = GaussianMLPPolicy(
            name="policy",
            env_spec=variant["env"].spec,
            hidden_sizes=(32,),
        )

        baseline = LinearFeatureBaseline(variant["env"].spec)

        bonus_evaluator = HashingBonusEvaluator(
            variant["env"].spec,
            variant["dim_key"],
        )

        algo = BonusTRPO(
            bonus_evaluator=bonus_evaluator,
            bonus_coeff=variant["bonus_coeff"],
            env=variant["env"],
            policy=policy,
            baseline=baseline,
            batch_size=variant["batch_size"],
            whole_paths=True,
            max_path_length=variant["max_path_length"],
            n_itr=N_ITR,
            discount=variant["discount"],
            step_size=variant["step_size"],
            plot=args.visualize and args.local,
            force_batch_sampler=True,
        )

        run_experiment_lite(
            algo.train(),
            n_parallel=1,
            snapshot_mode="all",
            seed=variant["seed"],
            plot=args.visualize and args.local,
            mode="local" if args.local else "ec2",
            exp_prefix=exp_prefix,
            terminate_machine=not args.debug
        )

        if args.debug:
            break  # Only run first experiment variant in debug mode
