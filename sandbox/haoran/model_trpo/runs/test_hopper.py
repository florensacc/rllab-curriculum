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
from sandbox.haoran.model_trpo.code.analytic_env import AnalyticEnv
from sandbox.haoran.model_trpo.code.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.haoran.model_trpo.code.deterministic_mlp_policy import DeterministicMLPPolicy
from sandbox.haoran.model_trpo.code.deterministic_value_gradient import DeterministicValueGradient
from sandbox.haoran.model_trpo.code.value_gradient import ValueGradient

from sandbox.haoran.model_trpo.code.hopper_env import HopperEnv
from sandbox.haoran.model_trpo.code.point_env import PointEnv


stub(globals())

# define running mode specific params -----------------------------------
exp_prefix = "model_trpo/" + os.path.basename(__file__).split('.')[0] # exp_xxx
mode = "local_test"
snapshot_mode = "all"
plot = False
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
    config.AWS_INSTANCE_TYPE = "g2.2xlarge"
    config.AWS_SPOT_PRICE = '0.5'
    config.DOCKER_IMAGE = "tsukuyomi2044/rllab_gpu"
    plot = False

# different training params ------------------------------------------
from rllab.misc.instrument import VariantGenerator, variant
class VG(VariantGenerator):
    @variant
    def seed(self):
        return [1]

    @variant
    def env(self):
        return ["hopper"]

    @variant
    def lr(self):
        return [0.1]

    @variant
    def fd_step(self):
        return [1e-3]

variants = VG().variants()
exp_names = []
for v in variants:
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

    # parameters --------------------------------------------
    n_itr = 100
    batch_size = 10000
    max_path_length = 500
    discount = 0.99
    lr = v["lr"]
    pause_for_plot = False
    fd_step = v["fd_step"]

    # environment
    if v["env"] == "hopper":
        wrapped_env = HopperEnv(use_full_state=True)
    elif v["env"] == "point":
        wrapped_env = PointEnv()
    else:
        raise NotImplementedError
    env = AnalyticEnv(wrapped_env, fd_step)

    # policy
    policy = GaussianMLPPolicy(
        env,
        hidden_sizes=[32,32],
        bound_output=True,
        init_std=0.1,
        learn_std=False, # should not learn std, other SVG formula breakdown
    )

    # baseline
    baseline = ZeroBaseline(env)

    # algorithm
    algo = ValueGradient(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        n_itr=n_itr,
        max_path_length=max_path_length,
        discount=discount,
        plot=plot,
        pause_for_plot=pause_for_plot,
        lr=lr,
    )


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
        seed=v["seed"],
        n_parallel=1, # we actually don't use parallel_sampler here
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
