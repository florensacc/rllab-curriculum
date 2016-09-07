from __future__ import print_function
from __future__ import absolute_import

from sandbox.haoran.hashing.bonus_trpo.algos.bonus_trpo import BonusTRPO
from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.hashing_bonus_evaluator import HashingBonusEvaluator
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.haoran.hashing.bonus_trpo.envs.atari import AtariEnv
from rllab import config
import sys,os

# stub(globals())

from rllab.misc.instrument import VariantGenerator, variant

"""
Fix to counting scheme. Fix config...
"""

exp_prefix = "bonus-trpo-atari/" + os.path.basename(__file__).split('.')[0] # exp_xxx
mode = "local_test"
n_parallel = 8
snapshot_mode = "last"
plot = False
use_gpu = False # should change conv_type and ~/.theanorc

if "ec2_m4" in mode:
    config.AWS_INSTANCE_TYPE = "m4.large"
    config.AWS_SPOT_PRICE = '1.5'
    config.DOCKER_IMAGE = "dementrock/rllab-shared"
    plot = False
elif "ec2_c4" in mode:
    config.AWS_INSTANCE_TYPE = "c4.large"
    config.AWS_SPOT_PRICE = '1.5'
    config.DOCKER_IMAGE = "dementrock/rllab-shared"
elif "ec2_g2" in mode:
    config.AWS_INSTANCE_TYPE = "g2.2xlarge"
    config.AWS_SPOT_PRICE = '1.5'
    config.DOCKER_IMAGE = "tsukuyomi2044/rllab_gpu"
    plot = False


# params ---------------------------------------
batch_size = 10000
clip_reward = True
max_path_length = 4500
n_itr = 100

class VG(VariantGenerator):
    @variant
    def seed(self):
        return [311, 411, 511, 611, 711]

    @variant
    def bonus_coeff(self):
        return [0.1, 0.01, 0.001,0]

    @variant
    def dim_key(self):
        return [64, 256]

    @variant
    def discount(self):
        return [0.99]

    @variant
    def game(self):
        return ["breakout"]


variants = VG().variants()


print("#Experiments: %d" % len(variants))

for v in variants:
    import datetime
    import dateutil.tz
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    exp_name = "alex_{time}_{game}".format(
        time=timestamp,
        game=v["game"],
    )
    if ("ec2" in mode) and (len(exp_name) > 64):
        print("Should not use experiment name with length %d > 64.\nThe experiment name is %s.\n Exit now."%(len(exp_name),exp_name))
        sys.exit(1)

    env = TfEnv(AtariEnv(game=v["game"], obs_type="ram"))
    policy = CategoricalMLPPolicy(env_spec=env.spec, hidden_sizes=(32, 32), name="policy")
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    bonus_baseline = LinearFeatureBaseline(env_spec=env.spec)
    bonus_evaluator = HashingBonusEvaluator(env_spec=env.spec, dim_key=v["dim_key"])
    algo = BonusTRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        bonus_evaluator=bonus_evaluator,
        bonus_baseline=bonus_baseline,
        extra_bonus_evaluator=None,
        bonus_coeff=v["bonus_coeff"],
        batch_size=batch_size,
        max_path_length=max_path_length,
        discount=v["discount"],
        n_itr=n_itr,
        clip_reward=clip_reward,
        plot=plot,
    )

    # run --------------------------------------------------
    if "local_docker" in mode:
        actual_mode = "local_docker"
    elif "local" in mode:
        actual_mode = "local"
    elif "ec2" in mode:
        actual_mode = "ec2"
    else:
        raise NotImplementedError


    run_experiment_lite(
        algo.train(),
        exp_prefix=exp_prefix,
        exp_name=exp_name,
        seed=v["seed"],
        n_parallel=n_parallel,
        snapshot_mode=snapshot_mode,
        mode=actual_mode,
        variant=v,
        use_gpu=use_gpu,
    )

    if "test" in mode:
        sys.exit(0)

if "local" not in mode:
    os.system("chmod 444 %s"%(__file__))
