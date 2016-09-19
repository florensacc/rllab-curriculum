"""
Test the save-load resetter on MLP policies
"""
from sandbox.rocky.tf.baselines.linear_feature_baseline import LinearFeatureBaseline
# from sandbox.rocky.tf.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from sandbox.rocky.tf.baselines.gaussian_conv_baseline import GaussianConvBaseline
from sandbox.rocky.tf.baselines.zero_baseline import ZeroBaseline
# from sandbox.rocky.tf.policies.categorical_conv_policy import CategoricalConvPolicy
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
# from sandbox.rocky.tf.policies.categorical_gru_policy import CategoricalGRUPolicy
from sandbox.rocky.tf.policies.categorical_ramdom_policy import CategoricalRandomPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp
from sandbox.rocky.tf.core.network import ConvNetwork
from sandbox.rocky.tf.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer

from sandbox.haoran.hashing.bonus_trpo.algos.bonus_trpo import BonusTRPO
from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.hashing_bonus_evaluator import HashingBonusEvaluator
from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.zero_bonus_evaluator import ZeroBonusEvaluator
from sandbox.haoran.hashing.bonus_trpo.envs.atari_env import AtariEnv
# from sandbox.haoran.hashing.bonus_trpo.resetter.atari_count_resetter import AtariCountResetter
from sandbox.haoran.hashing.bonus_trpo.resetter.atari_save_load_resetter import AtariSaveLoadResetter
from sandbox.haoran.hashing.bonus_trpo.misc.dqn import nips_dqn_args
from sandbox.haoran.myscripts.myutilities import get_time_stamp
from sandbox.haoran.ec2_info import instance_info, subnet_info

from rllab import config
from rllab.misc.instrument import stub, run_experiment_lite
import sys,os
import copy

stub(globals())
import tensorflow as tf

from rllab.misc.instrument import VariantGenerator, variant

exp_prefix = "bonus-trpo-atari/" + os.path.basename(__file__).split('.')[0] # exp_xxx
mode = "ec2"
ec2_instance = "c4.large"
subnet = "us-west-1c"

n_parallel = 1
snapshot_mode = "last"
plot = False
use_gpu = False # should change conv_type and ~/.theanorc
sync_s3_pkl = True


# params ---------------------------------------
max_path_length = 4500
discount = 0.99
n_itr = 200

clip_reward = True
obs_type = "ram"
record_image=False
record_rgb_image=False
record_ram=True
record_internal_state=False

dim_key = 64
bonus_form="1/sqrt(n)"
extra_dim_key = 1024
extra_bucket_sizes = [15485867, 15485917, 15485927, 15485933, 15485941, 15485959]

# restored_state_folder = '/tmp/restored_state'
restored_state_folder = None
avoid_life_lost = True


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [1, 111, 211, 311, 411, 511, 611, 711, 811, 911]

    # The environment seed does affect Atari games. For example, the result of breaking two blocks or one block with one hit at their intersection depends on the ramdom seed.
    @variant
    def env_seed(self):
        return [1, 111]

    @variant
    def bonus_coeff(self):
        return [0]

    @variant
    def game(self):
        return ["space_invaders", "breakout", "pong", "beam_rider", "qbert", "frostbite"]

    @variant
    def policy_type(self):
        return ["mlp"]

    @variant
    def batch_size(self):
        return [10000,20000]
variants = VG().variants()


print("#Experiments: %d" % len(variants))
for v in variants:
    exp_name = "alex_{time}_{game}_{obs_type}".format(
        time=get_time_stamp(),
        game=v["game"],
        obs_type=obs_type,
    )
    if ("ec2" in mode) and (len(exp_name) > 64):
        print("Should not use experiment name with length %d > 64.\nThe experiment name is %s.\n Exit now."%(len(exp_name),exp_name))
        sys.exit(1)

    resetter = AtariSaveLoadResetter(
        restored_state_folder=restored_state_folder,
        avoid_life_lost=avoid_life_lost,
    )
    env = TfEnv(
        AtariEnv(
            game=v["game"],
            seed=v["env_seed"],
            obs_type=obs_type,
            record_ram=record_ram,
            record_image=record_image,
            record_rgb_image=record_rgb_image,
            record_internal_state=record_internal_state,
            resetter=resetter,
            avoid_life_lost=avoid_life_lost,
        )
    )
    # policy = CategoricalConvPolicy(
    #     env_spec=env.spec,
    #     name="policy",
    #     **nips_dqn_args
    # )
    if v["policy_type"] == "mlp":
        policy = CategoricalMLPPolicy(env_spec=env.spec, hidden_sizes=(32, 32), name="policy")
        print('hhhhhhhhhhh')
    elif v["policy_type"] == "random":
        policy = CategoricalRandomPolicy(env_spec=env.spec)
    baseline = LinearFeatureBaseline(env_spec=env.spec)


    bonus_evaluator = ZeroBonusEvaluator()
    # extra_bonus_evaluator = HashingBonusEvaluator(
    #     env_spec=env.spec,
    #     dim_key=extra_dim_key,
    #     bucket_sizes=extra_bucket_sizes,
    #     log_prefix="Extra",
    # )
    extra_bonus_evaluator = ZeroBonusEvaluator()
    algo = BonusTRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        bonus_evaluator=bonus_evaluator,
        extra_bonus_evaluator=extra_bonus_evaluator,
        bonus_coeff=v["bonus_coeff"],
        batch_size=v["batch_size"],
        max_path_length=max_path_length,
        discount=discount,
        n_itr=n_itr,
        clip_reward=clip_reward,
        plot=plot,
        optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5)),
        force_batch_sampler=True,
    )

    # run --------------------------------------------------
    if "local_docker" in mode:
        actual_mode = "local_docker"
    elif "local" in mode:
        actual_mode = "local"
    elif "ec2" in mode:
        actual_mode = "ec2"
        # configure instance
        info = instance_info[ec2_instance]
        config.AWS_INSTANCE_TYPE = ec2_instance
        config.AWS_SPOT_PRICE = str(info["price"])
        n_parallel = int(info["vCPU"] /2)

        # choose subnet
        config.AWS_NETWORK_INTERFACES = [
            dict(
                SubnetId=subnet_info[subnet]["SubnetID"],
                Groups=subnet_info[subnet]["Groups"],
                DeviceIndex=0,
                AssociatePublicIpAddress=True,
            )
        ]
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
        plot=plot,
        sync_s3_pkl=sync_s3_pkl,
    )

    if "test" in mode:
        sys.exit(0)

if ("local" not in mode) and ("test" not in mode):
    os.system("chmod 444 %s"%(__file__))
