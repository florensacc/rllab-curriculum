# Continue exp-002b, with a larger batch size
# games: montezuma

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

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant

"""
Fix to counting scheme. Fix config...
"""

exp_prefix = "bonus-trpo-atari/" + os.path.basename(__file__).split('.')[0] # exp_xxx
mode = "ec2_c4_2x"
n_parallel = 8
snapshot_mode = "last"
plot = False
use_gpu = False # should change conv_type and ~/.theanorc

if "ec2_m4" in mode:
    config.AWS_INSTANCE_TYPE = "m4.large"
    config.AWS_SPOT_PRICE = '0.12'
    config.DOCKER_IMAGE = "dementrock/rllab-shared"
    plot = False
    n_parallel=1
elif "ec2_m4_x" in mode:
    config.AWS_INSTANCE_TYPE = "m4.xlarge"
    config.AWS_SPOT_PRICE = '0.24'
    config.DOCKER_IMAGE = "dementrock/rllab-shared"
    plot = False
    n_parallel=2
elif "ec2_m4_2x" in mode:
    config.AWS_INSTANCE_TYPE = "m4.2xlarge"
    config.AWS_SPOT_PRICE = '0.48'
    config.DOCKER_IMAGE = "dementrock/rllab-shared"
    plot = False
    n_parallel=4
elif "ec2_c4" in mode:
    config.AWS_INSTANCE_TYPE = "c4.large"
    config.AWS_SPOT_PRICE = '0.105'
    config.DOCKER_IMAGE = "dementrock/rllab-shared"
    plot = False
    n_parallel=1
elif "ec2_c4_x" in mode:
    config.AWS_INSTANCE_TYPE = "c4.xlarge"
    config.AWS_SPOT_PRICE = '0.209'
    config.DOCKER_IMAGE = "dementrock/rllab-shared"
    plot = False
    n_parallel=2
elif "ec2_c4_2x" in mode:
    config.AWS_INSTANCE_TYPE = "c4.2xlarge"
    config.AWS_SPOT_PRICE = '0.419'
    config.DOCKER_IMAGE = "dementrock/rllab-shared"
    plot = False
    n_parallel=4
elif "ec2_g2" in mode:
    config.AWS_INSTANCE_TYPE = "g2.2xlarge"
    config.AWS_SPOT_PRICE = '1.5'
    config.DOCKER_IMAGE = "tsukuyomi2044/rllab_gpu"
    plot = False
else:
    raise NotImplementedError


# params ---------------------------------------
batch_size = 100000
max_path_length = 4500
discount = 0.99
n_itr = 1000

clip_reward = True
extra_dim_key = 1024 
extra_bucket_sizes = [15485867, 15485917, 15485927, 15485933, 15485941, 15485959]


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [111, 211, 311, 411, 511, 611, 711]

    @variant
    def bonus_coeff(self):
        return [0.1, 0.01, 0.001, 0]

    @variant
    def dim_key(self):
        return [64]

    @variant
    def game(self):
        return ["montezuma_revenge"]

    @variant
    def bonus_form(self):
        return ["1/sqrt(n)"]

    @variant
    def death_ends_episode(self):
        return [False]

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

    env = TfEnv(AtariEnv(game=v["game"], obs_type="ram",death_ends_episode=v["death_ends_episode"]))
    policy = CategoricalMLPPolicy(env_spec=env.spec, hidden_sizes=(32, 32), name="policy")
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    bonus_baseline = LinearFeatureBaseline(env_spec=env.spec)
    bonus_evaluator = HashingBonusEvaluator(
        env_spec=env.spec, 
        dim_key=v["dim_key"],
        bonus_form=v["bonus_form"],
        log_prefix="",
    )
    extra_bonus_evaluator = HashingBonusEvaluator(
        env_spec=env.spec, 
        dim_key=extra_dim_key,
        bucket_sizes=extra_bucket_sizes,
        log_prefix="Extra",
    )
    algo = BonusTRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        bonus_evaluator=bonus_evaluator,
        extra_bonus_evaluator=extra_bonus_evaluator,
        bonus_baseline=bonus_baseline,
        bonus_coeff=v["bonus_coeff"],
        batch_size=batch_size,
        max_path_length=max_path_length,
        discount=discount,
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

if ("local" not in mode) and ("test" not in mode):
    os.system("chmod 444 %s"%(__file__))

