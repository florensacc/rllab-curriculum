# imports -----------------------------------------------------
""" baseline """
#from sandbox.haoran.parallel_trpo.linear_feature_baseline import ParallelLinearFeatureBaseline
from sandbox.sandy.parallel_trpo.gaussian_conv_baseline import ParallelGaussianConvBaseline

""" policy """
from rllab.policies.categorical_conv_policy import CategoricalConvPolicy
from sandbox.sandy.train.network_args import nips_dqn_args

""" optimizer """
from sandbox.sandy.parallel_trpo.conjugate_gradient_optimizer import ParallelConjugateGradientOptimizer

""" algorithm """
from sandbox.sandy.parallel_trpo.trpo import ParallelTRPO

""" environment """
#from sandbox.sandy.envs.atari_env_haoran_minimal import AtariEnvMinimal
#from sandbox.sandy.envs.atari_env_haoran import AtariEnv
from sandbox.sandy.envs.atari_env import AtariEnv
from rllab.envs.normalized_env import normalize

""" others """
from sandbox.sandy.misc.util import get_time_stamp
from sandbox.sandy.misc.ec2_info import instance_info, subnet_info
from rllab import config
from rllab.misc.instrument import stub, run_experiment_lite
import numpy as np
import sys,os
import copy

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant

# exp setup -----------------------------------------------------
exp_index = os.path.basename(__file__).split('.')[0] # exp_xxx
exp_prefix = "trpo-pong/" + exp_index
mode = "ec2"  # "local_docker"
ec2_instance = "c4.8xlarge"
price_multiplier = 3
subnet = "us-west-1a"
config.DOCKER_IMAGE = "tsukuyomi2044/rllab3:theano" # needs psutils

n_parallel = 2 # only for local exp
snapshot_mode = "gap"
snapshot_gap = 50
plot = False
use_gpu = False
sync_s3_pkl = True
sync_s3_log = True
config.USE_TF = False

if "local" in mode and sys.platform == "darwin":
    set_cpu_affinity = False
    cpu_assignments = None
    serial_compile = False
else:
    set_cpu_affinity = True
    cpu_assignments = None
    serial_compile = True

# variant params ---------------------------------------
class VG(VariantGenerator):
    @variant
    def seed(self):
        return [0, 100, 200, 300, 400]

    @variant
    def game(self):
        return ["pong"]

variants = VG().variants()

# test whether all game names are spelled correctly (comment out stub(globals) first)
# tested_games = []
# for v in variants:
#     game = v["game"]
#     if game not in tested_games:
#         env = AtariEnv(game=game)
#         print("Game %s tested"%(game))
#         tested_games.append(game)
# sys.exit(0)

print("#Experiments: %d" % len(variants))
for v in variants:
    # non-variant params ------------------------------
    # algo
    use_parallel = True
    seed = v["seed"]
    if mode == "local_test":
        batch_size = 500
    elif mode == "local_docker":
        batch_size = 10000
    else:
        batch_size = 100000
    max_path_length = 4500
    discount = 0.99
    n_itr = 500
    step_size = 0.01
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
    network_args = nips_dqn_args
    clip_reward = True  # Clip rewards to be from -1 to 1

    # env
    game = v["game"]
    env_seed = 1 # deterministic env
    frame_skip = 4
    max_start_nullops = 0
    img_width = 42
    img_height = 42
    n_last_screens = 4
    obs_type = "image"
    record_image = False
    record_rgb_image = False
    record_ram = False
    record_internal_state = False

    # other exp setup --------------------------------------
    exp_name = "{exp_index}_{time}_{game}".format(
        exp_index=exp_index,
        time=get_time_stamp(),
        game=game,
    )
    if ("ec2" in mode) and (len(exp_name) > 64):
        print("Should not use experiment name with length %d > 64.\nThe experiment name is %s.\n Exit now."%(len(exp_name),exp_name))
        sys.exit(1)

    if use_gpu:
        config.USE_GPU = True
        config.DOCKER_IMAGE = "dementrock/rllab3-shared-gpu"

    if "local_docker" in mode:
        actual_mode = "local_docker"
    elif "local" in mode:
        actual_mode = "local"
    elif "ec2" in mode:
        actual_mode = "ec2"
        # configure instance
        info = instance_info[ec2_instance]
        config.AWS_INSTANCE_TYPE = ec2_instance
        config.AWS_SPOT_PRICE = str(info["price"] * price_multiplier)
        if config.AWS_REGION_NAME == "us-west-1":
             config.AWS_IMAGE_ID = "ami-271b4847"  # Use Haoran's AWS image with his docker iamge

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
    #elif "kube" in mode:
    #    actual_mode = "lab_kube"
    #    info = instance_info[ec2_instance]
    #    n_parallel = int(info["vCPU"] /2)

    #    config.KUBE_DEFAULT_RESOURCES = {
    #        "requests": {
    #            "cpu": int(info["vCPU"]*0.75)
    #        }
    #    }
    #    config.KUBE_DEFAULT_NODE_SELECTOR = {
    #        "aws/type": ec2_instance
    #    }
    #    exp_prefix = exp_prefix.replace('/','-') # otherwise kube rejects
    else:
        raise NotImplementedError

    # construct objects ----------------------------------
    #env = normalize(AtariEnv('Pong-v3', force_reset=True))
    env = AtariEnv('Pong-v3', force_reset=True, seed=env_seed)
    #env = AtariEnvMinimal(
    #    game=game,
    #    seed=env_seed,
    #    img_width=img_width,
    #    img_height=img_height,
    #    obs_type=obs_type,
    #    record_ram=record_ram,
    #    record_image=record_image,
    #    record_rgb_image=record_rgb_image,
    #    record_internal_state=record_internal_state,
    #    frame_skip=frame_skip,
    #    max_start_nullops=max_start_nullops,
    #    correct_luminance=False,
    #    resize_with_scipy=True
    #)
    policy = CategoricalConvPolicy(
        env_spec=env.spec,
        name="policy",
        **network_args
    )

    # baseline
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
            batchsize=batch_size*10,
            normalize_inputs=True,
            normalize_outputs=True,
            **network_args_for_vf
        )
    )

    algo = ParallelTRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        max_path_length=max_path_length,
        discount=discount,
        n_itr=n_itr,
        clip_reward=clip_reward,
        plot=plot,
        optimizer_args=policy_opt_args,
        step_size=step_size,
        set_cpu_affinity=set_cpu_affinity,
        cpu_assignments=cpu_assignments,
        serial_compile=serial_compile,
        n_parallel=n_parallel,
    )

    if use_parallel:
        run_experiment_lite(
            algo.train(),
            exp_prefix=exp_prefix,
            exp_name=exp_name,
            seed=seed,
            snapshot_mode=snapshot_mode,
            snapshot_gap=snapshot_gap,
            mode=actual_mode,
            variant=v,
            use_gpu=use_gpu,
            plot=plot,
            sync_s3_pkl=sync_s3_pkl,
            sync_s3_log=sync_s3_log,
            sync_log_on_termination=True,
            sync_all_data_node_to_s3=True,
        )
    else:
        raise NotImplementedError

    if "test" in mode:
        sys.exit(0)

if ("local" not in mode) and ("test" not in mode):
    os.system("chmod 444 %s"%(__file__))
