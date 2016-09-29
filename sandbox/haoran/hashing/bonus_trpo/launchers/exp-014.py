"""
Use image observations. Can compare to exp-013.
Switch to Theano. Run on CPU.
"""
from rllab.baselines.gaussian_conv_baseline import GaussianConvBaseline
from rllab.baselines.zero_baseline import ZeroBaseline

from rllab.policies.categorical_conv_policy import CategoricalConvPolicy
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy

from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp
from rllab.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from rllab.optimizers.first_order_optimizer import FirstOrderOptimizer

from sandbox.haoran.hashing.bonus_trpo.algos.bonus_trpo_theano import BonusTRPO
from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.hashing_bonus_evaluator import HashingBonusEvaluator
from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.zero_bonus_evaluator import ZeroBonusEvaluator
from sandbox.haoran.hashing.bonus_trpo.envs.atari_env import AtariEnv
# from sandbox.haoran.hashing.bonus_trpo.resetter.atari_count_resetter import AtariCountResetter
from sandbox.haoran.hashing.bonus_trpo.misc.dqn_args_theano import trpo_dqn_args
from sandbox.haoran.myscripts.myutilities import get_time_stamp
from sandbox.haoran.ec2_info import instance_info, subnet_info

from rllab import config
from rllab.misc.instrument import stub, run_experiment_lite
import sys,os
import copy

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant

exp_prefix = "bonus-trpo-atari/" + os.path.basename(__file__).split('.')[0] # exp_xxx
mode = "kube_test"
ec2_instance = "c4.8xlarge"
subnet = "us-west-1a"

n_parallel = 4
snapshot_mode = "last"
plot = False
use_gpu = False # should change conv_type and ~/.theanorc
sync_s3_pkl = True
config.USE_TF = False

# params ---------------------------------------
batch_size = 10000
max_path_length = 4500
discount = 0.99
n_itr = 1000
force_batch_sampler = True
cg_args = dict(
    cg_iters=10,
    reg_coeff=1e-3,
    subsample_factor=0.1,
    max_backtracks=15,
    backtrack_ratio=0.8,
    accept_violation=False,
    hvp_approach=FiniteDifferenceHvp(base_eps=1e-5),
    num_slices=10,
)
step_size = 0.01
network_args = trpo_dqn_args

img_width=42
img_height=42
clip_reward = True
obs_type = "image"
record_image=False
record_rgb_image=False
record_ram=True
record_internal_state=False

dim_key = 64
bonus_form="1/sqrt(n)"
extra_dim_key = 1024
extra_bucket_sizes = [15485867, 15485917, 15485927, 15485933, 15485941, 15485959]

baseline_opt_type = "sgd"

class VG(VariantGenerator):
    @variant
    def seed(self):
        return [111, 211, 311]

    @variant
    def bonus_coeff(self):
        return [0]

    @variant
    def game(self):
        return ["beam_rider"]
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

    resetter = None
    env = AtariEnv(
            game=v["game"],
            seed=v["seed"],
            img_width=img_width,
            img_height=img_height,
            obs_type=obs_type,
            record_ram=record_ram,
            record_image=record_image,
            record_rgb_image=record_rgb_image,
            record_internal_state=record_internal_state,
            resetter=resetter,
        )
    policy = CategoricalConvPolicy(
        env_spec=env.spec,
        name="policy",
        **network_args
    )


    baseline = ZeroBaseline(env_spec=env.spec)
    # network_args_for_vf = copy.deepcopy(network_args)
    # network_args_for_vf.pop("output_nonlinearity")
    # if baseline_opt_type == "bfgs":
    #     baseline = GaussianConvBaseline(
    #         env_spec=env.spec,
    #         regressor_args = dict(
    #             optimizer = PenaltyLbfgsOptimizer(
    #                 max_opt_itr=20,
    #                 max_penalty_itr=2,
    #             ),
    #             use_trust_region=True,
    #             step_size=0.01,
    #             init_std=1.0,
    #             **network_args_for_vf
    #         )
    #     )
    # elif baseline_opt_type == "sgd":
    #     baseline = GaussianConvBaseline(
    #         env_spec=env.spec,
    #         regressor_args = dict(
    #             optimizer=FirstOrderOptimizer(
    #                 max_epochs=1,
    #                 batch_size=1000,
    #                 learning_rate=1e-3,
    #             ),
    #             use_trust_region=False,
    #             init_std=1.0,
    #             **network_args_for_vf
    #         )
    #     )
    # else:
    #     raise NotImplementedError


    # bonus_evaluator = HashingBonusEvaluator(
    #     env_spec=env.spec,
    #     dim_key=dim_key,
    #     bonus_form=bonus_form,
    #     log_prefix="",
    # )
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
        batch_size=batch_size,
        max_path_length=max_path_length,
        discount=discount,
        n_itr=n_itr,
        clip_reward=clip_reward,
        plot=plot,
        optimizer=ConjugateGradientOptimizer(**cg_args),
        step_size=step_size,
        force_batch_sampler=force_batch_sampler,
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
    elif "kube" in mode:
        actual_mode = "lab_kube"
        info = instance_info[ec2_instance]
        n_parallel = int(info["vCPU"] /2)

        config.KUBE_DEFAULT_RESOURCES = {
            "requests": {
                "cpu": n_parallel
            }
        }
        config.KUBE_DEFAULT_NODE_SELECTOR = {
            "aws/type": ec2_instance
        }
        exp_prefix = exp_prefix.replace('/','-') # otherwise kube rejects
    else:
        raise NotImplementedError

    if use_gpu:
        config.USE_GPU = True
        config.DOCKER_IMAGE = "dementrock/rllab3-shared-gpu"

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
        sync_log_on_termination=True,
    )

    if "test" in mode:
        sys.exit(0)

if ("local" not in mode) and ("test" not in mode):
    os.system("chmod 444 %s"%(__file__))
