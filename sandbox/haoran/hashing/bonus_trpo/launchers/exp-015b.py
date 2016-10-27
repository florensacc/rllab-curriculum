"""
Tune parameters on Breakout
1. Smaller kl
2. clip baseline prediction
"""
""" baseline """
from sandbox.adam.parallel.gaussian_conv_baseline import ParallelGaussianConvBaseline
from sandbox.adam.parallel.parallel_nn_feature_linear_baseline import ParallelNNFeatureLinearBaseline

""" policy """
from rllab.policies.categorical_conv_policy import CategoricalConvPolicy
from sandbox.haoran.hashing.bonus_trpo.misc.dqn_args_theano import trpo_dqn_args,nips_dqn_args

""" optimizer """
from sandbox.haoran.parallel_trpo.conjugate_gradient_optimizer import ParallelConjugateGradientOptimizer

""" algorithm """
from sandbox.haoran.parallel_trpo.trpo import ParallelTRPO

""" environment """
from sandbox.haoran.hashing.bonus_trpo.envs.atari_env import AtariEnv

""" resetter """
# from sandbox.haoran.hashing.bonus_trpo.resetter.atari_count_resetter import AtariCountResetter

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
mode = "kube"
ec2_instance = "c4.8xlarge"
subnet = "us-west-1a"
config.DOCKER_IMAGE = "tsukuyomi2044/rllab3" # needs psutils

n_parallel = 4
snapshot_mode = "last"
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
use_parallel = True
batch_size = 50000
max_path_length = 4500
discount = 0.99
n_itr = 1000
step_size = 0.001
policy_opt_args = dict(
    name="pi_opt",
    cg_iters=10,
    reg_coeff=1e-3,
    subsample_factor=0.1,
    max_backtracks=15,
    backtrack_ratio=0.8,
    accept_violation=False,
    hvp_approach=None,
    num_slices=1, # reduces memory requirement
)

# env
network_args = nips_dqn_args
img_width=84
img_height=84
clip_reward = True
obs_type = "image"
record_image=False
record_rgb_image=False
record_ram=True
record_internal_state=False

# bonus
dim_key = 64
bonus_form="1/sqrt(n)"
extra_dim_key = 1024
extra_bucket_sizes = [15485867, 15485917, 15485927, 15485933, 15485941, 15485959]


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [0,100,200]

    @variant
    def bonus_coeff(self):
        return [0]

    @variant
    def baseline_type_opt(self):
        return [
            # ["conv","cg"],
            ["nn_feature_linear",""],
        ]

    @variant
    def game(self):
        return ["breakout"]
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

    baseline_type, baseline_opt = v["baseline_type_opt"]
    if baseline_type == "nn_feature_linear":
        baseline = ParallelNNFeatureLinearBaseline(
            env_spec=env.spec,
            policy=policy,
            nn_feature_power=1,
            t_power=3,
            prediction_clip=1000,
        )
    elif baseline_type == "conv":
        network_args_for_vf = copy.deepcopy(network_args)
        network_args_for_vf.pop("output_nonlinearity")
        baseline = ParallelGaussianConvBaseline(
            env_spec=env.spec,
            regressor_args = dict(
                optimizer=ParallelConjugateGradientOptimizer(
                    subsample_factor=0.1,
                    cg_iters=10,
                    name="vf_opt",
                ),
                use_trust_region=True,
                step_size=0.01,
                batchsize=batch_size,
                normalize_inputs=True,
                normalize_outputs=True,
                **network_args_for_vf
            )
        )
    else:
        raise NotImplementedError


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
    )

    if use_gpu:
        config.USE_GPU = True
        config.DOCKER_IMAGE = "dementrock/rllab3-shared-gpu"

    if use_parallel:
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
        raise NotImplementedError

    if "test" in mode:
        sys.exit(0)

if ("local" not in mode) and ("test" not in mode):
    os.system("chmod 444 %s"%(__file__))
