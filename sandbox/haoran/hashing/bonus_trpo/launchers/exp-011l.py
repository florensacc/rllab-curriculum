"""
Continue exp-011k. Find out why new_state_count seems too small. Do exact comparison with the non-parallel version.
"""

""" baseline """
from sandbox.adam.parallel.gaussian_conv_baseline import ParallelGaussianConvBaseline
from sandbox.adam.parallel.parallel_nn_feature_linear_baseline import ParallelNNFeatureLinearBaseline
from sandbox.haoran.parallel_trpo.linear_feature_baseline import ParallelLinearFeatureBaseline
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline

""" policy """
from rllab.policies.categorical_conv_policy import CategoricalConvPolicy
from sandbox.haoran.hashing.bonus_trpo.misc.dqn_args_theano import trpo_dqn_args,nips_dqn_args
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy

""" optimizer """
from sandbox.haoran.parallel_trpo.conjugate_gradient_optimizer import ParallelConjugateGradientOptimizer
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer

""" algorithm """
from sandbox.haoran.parallel_trpo.trpo import ParallelTRPO
from sandbox.haoran.hashing.bonus_trpo.algos.bonus_trpo_theano import BonusTRPO
from sandbox.adam.modified_sampler.batch_sampler import BatchSampler

""" environment """
from sandbox.haoran.hashing.bonus_trpo.envs.atari_env import AtariEnv

""" resetter """
# from sandbox.haoran.hashing.bonus_trpo.resetter.atari_count_resetter import AtariCountResetter

""" bonus """
from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.ale_hashing_bonus_evaluator import ALEHashingBonusEvaluator
from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.hash.sim_hash import SimHash
from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.hash.ale_hacky_hash_v2 import ALEHackyHashV2
from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.preprocessor.slicing_preprocessor import SlicingPreprocessor
from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.preprocessor.image_vectorize_preprocessor import ImageVectorizePreprocessor

""" others """
from sandbox.haoran.myscripts.myutilities import get_time_stamp
from sandbox.haoran.ec2_info import instance_info, subnet_info
from rllab import config
from rllab.misc.instrument import stub, run_experiment_lite
import sys,os
import copy

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant

exp_prefix = "bonus-trpo-atari/" + os.path.basename(__file__).split('.')[0] # exp_xxx
mode = "local_test"
ec2_instance = "c4.8xlarge"
subnet = "us-west-1a"
config.DOCKER_IMAGE = "tsukuyomi2044/rllab3" # needs psutils

n_parallel = 2
snapshot_mode = "none"
plot = False
use_gpu = False # should change conv_type and ~/.theanorc
sync_s3_pkl = True
config.USE_TF = False

if "local" in mode and sys.platform == "darwin":
    set_cpu_affinity = False
    cpu_assignments = None
    serial_compile = False
else:
    set_cpu_affinity = True
    cpu_assignments = None
    serial_compile = True

# params ---------------------------------------
# algo
use_parallel = False
if "test" in mode:
    batch_size = 1000
else:
    batch_size = 100000
max_path_length = 4500
discount = 0.99
n_itr = 1000
step_size = 0.01
policy_opt_args = dict(
    name="pi_opt",
    cg_iters=10,
    reg_coeff=1e-5,
    subsample_factor=1.0,
    max_backtracks=15,
    backtrack_ratio=0.8,
    accept_violation=False,
    hvp_approach=None,
    num_slices=1, # reduces memory requirement
)

# env
img_width=84
img_height=84
n_last_screens=1
n_last_rams=1
clip_reward = True
obs_type = "ram"
record_image=True
record_rgb_image=False
record_ram=True
record_internal_state=False

# bonus
count_target = "images"
bonus_form="1/sqrt(n)"
bucket_sizes = [15485867, 15485917, 15485927, 15485933, 15485941, 15485959]


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [0,100,200,300,400,500,600,700,800,900]

    @variant
    def bonus_coeff(self):
        return [0.1]

    @variant
    def dim_key(self):
        return [256, 512]

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

    resetter = None
    if count_target == "observations" and obs_type == "image":
        total_pixels=img_width * img_height
        state_preprocessor = SlicingPreprocessor(
            input_dim=total_pixels * n_last_screens,
            start=total_pixels * (n_last_screens - 1),
            stop=total_pixels * n_last_screens,
            step=1,
        )
    elif count_target == "images":
        state_preprocessor = ImageVectorizePreprocessor(
            n_channel=n_last_screens,
            width=img_width,
            height=img_height,
        )
    elif count_target == "ram_states":
        state_preprocessor = None
    else:
        raise NotImplementedError

    _hash = SimHash(
        item_dim=state_preprocessor.get_output_dim(), # get around stub
        dim_key=v["dim_key"],
        bucket_sizes=bucket_sizes,
        parallel=use_parallel,
    )
    bonus_evaluator = ALEHashingBonusEvaluator(
        log_prefix="",
        state_dim=state_preprocessor.get_output_dim(),
        state_preprocessor=state_preprocessor,
        hash=_hash,
        bonus_form=bonus_form,
        count_target=count_target,
        parallel=use_parallel,
    )
    # extra_hash = ALEHackyHashV2(
    #     item_dim=128,
    #     game=v["game"],
    #     parallel=use_parallel,
    # )
    # extra_bonus_evaluator = ALEHashingBonusEvaluator(
    #     log_prefix="Extra",
    #     state_dim=128,
    #     state_preprocessor=None,
    #     hash=extra_hash,
    #     bonus_form=bonus_form,
    #     count_target="ram_states",
    #     parallel=use_parallel,
    # )
    extra_bonus_evaluator = None

    env = AtariEnv(
        game=v["game"],
        seed=v["seed"],
        img_width=img_width,
        img_height=img_height,
        n_last_screens=n_last_screens,
        n_last_rams=n_last_rams,
        obs_type=obs_type,
        record_ram=record_ram,
        record_image=record_image,
        record_rgb_image=record_rgb_image,
        record_internal_state=record_internal_state,
        resetter=resetter,
    )
    policy = CategoricalMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(32,32),
    )

    if use_parallel:
        baseline = ParallelLinearFeatureBaseline(env_spec=env.spec)

        algo = ParallelTRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=batch_size,
            max_path_length=max_path_length,
            discount=discount,
            n_itr=n_itr,
            plot=plot,
            optimizer_args=policy_opt_args,
            step_size=step_size,
            set_cpu_affinity=set_cpu_affinity,
            cpu_assignments=cpu_assignments,
            serial_compile=serial_compile,
            n_parallel=n_parallel,
            bonus_evaluator=bonus_evaluator,
            extra_bonus_evaluator=extra_bonus_evaluator,
            bonus_coeff=v["bonus_coeff"],
            clip_reward=clip_reward,
        )

        run_experiment_lite(
            algo.train(),
            exp_prefix=exp_prefix,
            exp_name=exp_name,
            seed=v["seed"],
            snapshot_mode=snapshot_mode,
            mode=actual_mode,
            variant=v,
            use_gpu=use_gpu,
            plot=plot,
            sync_s3_pkl=sync_s3_pkl,
            sync_log_on_termination=True,
            sync_all_data_node_to_s3=True,
        )
    else:
        baseline = LinearFeatureBaseline(env_spec=env.spec)

        policy_opt_args.pop("name")
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
            optimizer_args=policy_opt_args,
            step_size=step_size,
            sampler_cls=BatchSampler
        )

        run_experiment_lite(
            algo.train(),
            script="sandbox/haoran/parallel_trpo/run_experiment_lite.py",
            exp_prefix=exp_prefix,
            exp_name=exp_name,
            seed=v["seed"],
            snapshot_mode=snapshot_mode,
            mode=actual_mode,
            variant=v,
            use_gpu=use_gpu,
            plot=plot,
            sync_s3_pkl=sync_s3_pkl,
            sync_log_on_termination=True,
            sync_all_data_node_to_s3=True,
            n_parallel=n_parallel,
        )

    if "test" in mode:
        sys.exit(0)

if ("local" not in mode) and ("test" not in mode):
    os.system("chmod 444 %s"%(__file__))
