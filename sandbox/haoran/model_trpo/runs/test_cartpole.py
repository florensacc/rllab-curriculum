# Test the deterministic value gradient method on Hopper

from __future__ import print_function
from __future__ import absolute_import

import sys,os
sys.path.append('.')
import numpy as np
import theano
import json

from rllab import config
os.path.join(config.PROJECT_PATH)
from rllab.misc.instrument import stub, run_experiment_lite

from rllab.baselines.zero_baseline import ZeroBaseline

# policies
from sandbox.haoran.model_trpo.code.analytic_env import AnalyticEnv
from sandbox.haoran.model_trpo.code.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.haoran.model_trpo.code.deterministic_mlp_policy import DeterministicMLPPolicy

# algorithms
from sandbox.haoran.model_trpo.code.deterministic_value_gradient import DeterministicValueGradient
from sandbox.haoran.model_trpo.code.value_gradient import ValueGradient
from rllab.algos.vpg import VPG
from rllab.algos.trpo import TRPO
from sandbox.haoran.model_trpo.code.natural_value_gradient import NVG

# environments
from sandbox.haoran.model_trpo.code.hopper_env import HopperEnv
from sandbox.haoran.model_trpo.code.swimmer_env import SwimmerEnv
from sandbox.haoran.model_trpo.code.point_env import PointEnv
from sandbox.haoran.model_trpo.code.analytic_cartpole_env import AnalyticCartpoleEnv
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize

from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer


stub(globals())

# define running mode specific params -----------------------------------
exp_prefix = "model_trpo/" + os.path.basename(__file__).split('.')[0] # exp_xxx
mode = "ec2_m4_2x"
snapshot_mode = "last"
plot = False
pause_for_plot = False
use_gpu = False # should change conv_type and ~/.theanorc

# config.DOCKER_IMAGE = 'tsukuyomi2044/rllab'
if "ec2_cpu" in mode:
    config.AWS_INSTANCE_TYPE = "m4.large"
    config.AWS_SPOT_PRICE = '0.1'
    config.DOCKER_IMAGE = "dementrock/rllab-shared"
    plot = False
elif "ec2_c4" in mode:
    config.AWS_INSTANCE_TYPE = "c4.large"
    config.AWS_SPOT_PRICE = '1.5'
    config.DOCKER_IMAGE = "dementrock/rllab-shared"
elif "ec2_gpu" in mode:
    config.AWS_INSTANcE_TYPE = "g2.2xlarge"
    config.AWS_SPOT_PRICE = '0.5'
    config.DOCKER_IMAGE = "tsukuyomi2044/rllab_gpu"
    plot = False
elif "ec2_m4_2x" in mode:
    config.AWS_INSTANCE_TYPE = "m4.2xlarge"
    config.AWS_SPOT_PRICE = '0.48'
else:
    raise NotImplementedError

# different training params ------------------------------------------
from rllab.misc.instrument import VariantGenerator, variant
class VG(VariantGenerator):
    @variant
    def seed(self):
        return [1]

    @variant
    def env(self):
        return ["analytic_cartpole"]

    @variant
    def lr(self):
        return [0.001]


variants = VG().variants()
exp_names = []
for v in variants:
    # parameters --------------------------------------------
    ALGO = "trpo"
    n_itr = 100
    batch_size = 40000
    max_path_length = 100
    discount = 0.99
    n_parallel = 4

    step_size = 0.01
    init_std = 0.5
    normalize_env = False
    center_adv = False
    learn_std = False
    lr = v["lr"]
    fd_step = 0
    seed = v["seed"]

    # define the exp_name (log folder name) -------------------
    import datetime
    import dateutil.tz
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    exp_name = "alex_{time}_{env}".format(
        time=timestamp,
        env=v["env"],
    )
    exp_names.append(exp_name)
    if ("ec2" in mode) and (len(exp_name) > 64):
        print("Should not use experiment name with length %d > 64.\nThe experiment name is %s.\n Exit now."%(len(exp_name),exp_name))
        sys.exit(1)

    # environment
    if v["env"] == "hopper":
        env = HopperEnv(use_full_state=True)
    elif v["env"] == "swimmer":
        env = SwimmerEnv(use_full_state=False)
    elif v["env"] == "point":
        env = PointEnv()
    elif v["env"] == "analytic_cartpole":
        env = AnalyticCartpoleEnv()
    elif v["env"] == "cartpole":
        env = CartpoleEnv()
    else:
        raise NotImplementedError
    if normalize_env:
        env = normalize(env)

    # policy
    policy = GaussianMLPPolicy(
        env,
        hidden_sizes=[32,32],
        bound_output=False,
        init_std=init_std,
        learn_std=learn_std, # should not learn std, other SVG formula breakdown
    )

    # baseline
    baseline = ZeroBaseline(env)

    # algorithm
    algo_shared_params = dict(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        n_itr=n_itr,
        max_path_length=max_path_length,
        discount=discount,
        plot=plot,
        pause_for_plot=pause_for_plot,
        center_adv=center_adv,
        fd_step=fd_step,
    )
    if ALGO == "value_gradient":
        algo = ValueGradient(
            optimizer_args = dict(
                learning_rate=lr,
                max_epochs=1,
                batch_size=None,
            ),
            **algo_shared_params
        )
    elif ALGO == "vpg":
        algo = VPG(
            optimizer_args = dict(
                learning_rate=lr,
                max_epochs=1,
                batch_size=None,
            ),
            **algo_shared_params
        )
    elif ALGO == "trpo":
        optimizer = ConjugateGradientOptimizer(
            cg_iters=10,
            reg_coeff=1e-3,
            subsample_factor=0.5,
            backtrack_ratio=0.95,
            max_backtracks=50,

        )
        algo = TRPO(
            step_size=step_size,
            optimizer=optimizer,
            **algo_shared_params
        )
    elif ALGO == "nvg":
        optimizer = ConjugateGradientOptimizer(
            cg_iters=10,
            reg_coeff=1e-3,
            subsample_factor=0.5,
            backtrack_ratio=0.95,
            max_backtracks=50,
        )
        algo = NVG(
            step_size=step_size,
            optimizer=optimizer,
            **algo_shared_params
        )
    else:
        raise NotImplementedError

    # run --------------------------------------------------
    terminate_machine = "test" not in mode
    if "local_docker" in mode:
        actual_mode = "local_docker"
    elif "local" in mode:
        actual_mode = "local"
    elif "ec2" in mode:
        actual_mode = "ec2"
    else:
        raise NotImplementedError

    run_experiment_lite(
        stub_method_call=algo.train(),
        exp_prefix=exp_prefix,
        exp_name=exp_name,
        seed=seed,
        n_parallel=n_parallel, # we actually don't use parallel_sampler here
        snapshot_mode=snapshot_mode,
        mode=actual_mode,
        variant=v,
        terminate_machine=terminate_machine,
        use_gpu=use_gpu,
        plot=plot,
    )
    if "test" in mode:
        sys.exit(0)

# logging -------------------------------------------------------------
# make the current script read-only to avoid accidental changes after ec2 runs
if "local" not in mode:
    os.system("chmod 444 %s"%(__file__))
