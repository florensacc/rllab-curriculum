"""
Repeat exp-016b, but with frame_skip = 8 and the best params
"""
from sandbox.rocky.tf.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.tf.policies.categorical_ramdom_policy import CategoricalRandomPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp
from sandbox.rocky.tf.core.network import ConvNetwork
from sandbox.rocky.tf.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer

from sandbox.haoran.hashing.bonus_trpo.algos.bonus_trpo import BonusTRPO
from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.ale_hashing_bonus_evaluator import ALEHashingBonusEvaluator
from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.zero_bonus_evaluator import ZeroBonusEvaluator
from sandbox.haoran.hashing.bonus_trpo.envs.atari_env import AtariEnv
from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.hash.sim_hash import SimHash
from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.hash.ale_hacky_hash_v2 import ALEHackyHashV2
from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.preprocessor.image_vectorize_preprocessor import ImageVectorizePreprocessor
from sandbox.haoran.hashing.bonus_trpo.resetter.atari_save_load_resetter import AtariSaveLoadResetter
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
mode = "kube"
ec2_instance = "c4.8xlarge"
subnet = "us-west-1c"

n_parallel = 1
snapshot_mode = "last"
snapshot_gap = 50
store_paths = False
plot = False
use_gpu = False # should change conv_type and ~/.theanorc
sync_s3_pkl = True
config.USE_TF = True
frame_skip = True

# params ---------------------------------------
if "test" in mode:
    batch_size = 1000 # careful about memory usage
else:
    batch_size = 50000
max_path_length = 4500
discount = 0.99
n_itr = 1000

env_seed = 1
clip_reward = True
obs_type = "ram"
n_last_rams = 1
n_last_screens = 1
record_image=True
record_rgb_image=False
record_ram=True
record_internal_state=False

count_target = "ram_states"
bonus_form="1/sqrt(n)"
# bucket_sizes = [15485867, 15485917, 15485927, 15485933, 15485941, 15485959] #90M
bucket_sizes = None

step_size = 0.01

restored_state_folder = None
avoid_life_lost = False


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [0,100,200,300,400,500,600,700,800,900]


    @variant
    def bonus_coeff(self):
        return [0.01]

    @variant
    def game(self):
        return ["montezuma_revenge"]

    @variant
    def policy_type(self):
        return ["mlp"]

    @variant
    def dim_key(self):
        return [256]

    @variant
    def resetter(self):
        return ["none"]
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

    if v["resetter"] == "SL":
        resetter = AtariSaveLoadResetter(
            restored_state_folder=restored_state_folder,
            avoid_life_lost=avoid_life_lost,
        )
    else:
        resetter = None
    env = TfEnv(
        AtariEnv(
            game=v["game"],
            seed=env_seed,
            obs_type=obs_type,
            n_last_screens=n_last_screens,
            n_last_rams=n_last_rams,
            record_ram=record_ram,
            record_image=record_image,
            record_rgb_image=record_rgb_image,
            record_internal_state=record_internal_state,
            resetter=resetter,
            avoid_life_lost=avoid_life_lost,
            frame_skip=frame_skip,
        )
    )
    if v["policy_type"] == "mlp":
        policy = CategoricalMLPPolicy(env_spec=env.spec, hidden_sizes=(32, 32), name="policy")
    elif v["policy_type"] == "random":
        policy = CategoricalRandomPolicy(env_spec=env.spec)
    baseline = LinearFeatureBaseline(env_spec=env.spec)

    if count_target == "images":
        state_preprocessor = ImageVectorizePreprocessor(
            n_channel=n_last_screens,
            width=84,
            height=84,
            slices=[None,None,None],
        )
    elif count_target == "ram_states":
        state_preprocessor = ImageVectorizePreprocessor(
            n_channel=1,
            width=128,
            height=1,
        )
    else:
        raise NotImplementedError

    _hash = SimHash(
        item_dim=state_preprocessor.get_output_dim(), # get around stub
        dim_key=v["dim_key"],
        bucket_sizes=bucket_sizes,
    )
    bonus_evaluator = ALEHashingBonusEvaluator(
        log_prefix="",
        state_dim=state_preprocessor.get_output_dim(),
        state_preprocessor=state_preprocessor,
        hash=_hash,
        bonus_form=bonus_form,
        count_target=count_target,
    )
    extra_hash = ALEHackyHashV2(
        item_dim=128,
        game=v["game"],
    )
    extra_bonus_evaluator = ALEHashingBonusEvaluator(
        log_prefix="Extra",
        state_dim=128,
        state_preprocessor=state_preprocessor,
        hash=extra_hash,
        bonus_form=bonus_form,
        count_target="ram_states",
    )

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
        optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5)),
        force_batch_sampler=True,
        step_size=step_size,
        store_paths=store_paths,
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


    run_experiment_lite(
        algo.train(),
        exp_prefix=exp_prefix,
        exp_name=exp_name,
        seed=v["seed"],
        n_parallel=n_parallel,
        snapshot_mode=snapshot_mode,
        snapshot_gap=snapshot_gap,
        mode=actual_mode,
        variant=v,
        use_gpu=use_gpu,
        plot=plot,
        sync_s3_pkl=sync_s3_pkl,
        terminate_machine=("test" not in mode),
    )

    if "test" in mode:
        sys.exit(0)

if ("local" not in mode) and ("test" not in mode):
    os.system("chmod 444 %s"%(__file__))
