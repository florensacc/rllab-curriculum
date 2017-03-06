import numpy as np
import os
import os.path as osp
import pickle
import sys

""" experiment """
from sandbox.sandy.adversarial.adv_experiment import AdvSleeperExperiment

""" others """
from sandbox.sandy.misc.util import get_time_stamp, to_iterable
from sandbox.sandy.misc.ec2_info import instance_info, subnet_info
from rllab import config
from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant

# exp setup -----------------------------------------------------
exp_index = os.path.basename(__file__).split('.')[0] # exp_xxx
#mode = "ec2_gpu"
#mode = "local_docker_gpu_test"
#mode = "local_gpu_test"
mode = "ec2"
ec2_instance = "c4.8xlarge"
#ec2_instance = "p2.xlarge"
price_multiplier = 10
subnet = "us-west-2a"
config.DOCKER_IMAGE = "tsukuyomi2044/rllab3:theano" # needs psutils

# adversary setup -----------------------------------------------
adversary_algo = 'fgsm'
adversary_algo_param_names = ['norm', 'fgsm_eps', 'policy_adv', 'policy_rollout', 'k']  # Levels of all_output_h5
base_model_dir = 'sandbox/sandy/adversarial/trained_models_recurrent/'
if 'docker' in mode or 'ec2' in mode:
    model_dir = osp.join(config.DOCKER_CODE_DIR, base_model_dir)
else:
    model_dir = osp.join('/home/shhuang/src/rllab-private/', base_model_dir)

across_rollout = True  # Perturb when possible across the whole rollout, rather
                       # than individually at specific timesteps
threshold_perf = 0.80
threshold_n = 3
seed = 1  # TODO: Try a few different seeds
obs_min = 0
obs_max = 1
N = 1  # Because score doesn't seem to change when I change the seed
if 'test' in mode:
    N = 1
num_ts = 100

save_rollouts = False  # Set to True if you want to save and visualize rollouts
                       # Recommend setting this to False if there are a lot of runs,
                       # because saving rollouts takes up a *lot* of memory
test_transfer = False   # Whether to use adversarial examples generated on
                        # one model to attack a different model

use_gpu = False
plot = False
config.USE_TF = False

if subnet.startswith("us-west-1"):
    config.AWS_REGION_NAME="us-west-1"
elif subnet.startswith("us-west-2"):
    config.AWS_REGION_NAME="us-west-2"
else:
    raise NotImplementedError

config.AWS_KEY_NAME = config.ALL_REGION_AWS_KEY_NAMES[config.AWS_REGION_NAME]

if "ec2" in mode:
    info = instance_info[ec2_instance]
    config.AWS_INSTANCE_TYPE = ec2_instance
    config.AWS_SPOT_PRICE = str(info["price"] * price_multiplier)
    if config.AWS_REGION_NAME == "us-west-1":
        config.AWS_IMAGE_ID = "ami-271b4847"  # Use Haoran's AWS image with his docker iamge
    else:
        config.AWS_IMAGE_ID = "ami-7ea27a1e"

    # choose subnet
    config.AWS_NETWORK_INTERFACES = [
        dict(
            SubnetId=subnet_info[subnet]["SubnetID"],
            Groups=subnet_info[subnet]["Groups"],
            DeviceIndex=0,
            AssociatePublicIpAddress=True,
        )
    ]


if "ec2_gpu" in mode or "docker_gpu" in mode:
    config.DOCKER_IMAGE = "shhuang/rllab-gpu"
    if subnet.startswith("us-west-1"):
        config.AWS_IMAGE_ID = "ami-931a51f3"
    elif subnet.startswith("us-west-2"):
        config.AWS_IMAGE_ID = "ami-9af95dfa"
    else:
        raise NotImplementedError
    config.AWS_KEY_NAME = config.ALL_REGION_AWS_KEY_NAMES[config.AWS_REGION_NAME]

# variant params ---------------------------------------
class VG(VariantGenerator):
    @variant
    def game(self):
        return ['chopper', 'pong', 'seaquest', 'space']

    @variant
    def norm(self):
        return ['l-inf']  # Only testing on l-inf for now

    @variant
    def fgsm_eps(self):
        #return [[0.0005, 0.001, 0.002, 0.004, 0.008]]
        return [[0.008, 0.004, 0.002, 0.001, 0.0005]]

    @variant
    def experiment(self):
        return ['async-rl_exp038', 'async-rl_exp037'] # Format: algo-name_exp-index

    @variant
    def k(self):
        return [1, 2, 3, 4, 0]

    @variant
    def stepsize_schedule(self):
        return [[0.5]]

variants = VG().variants()

print("#Experiments: %d" % len(variants))
for v in variants:
    game = v['game']
    norm = v['norm']
    fgsm_eps = v['fgsm_eps']
    experiment = v['experiment']
    exp_prefix = "adv-sleeper-rollouts/" + exp_index
    k = v['k']
    k_init_lambda = []
    for val in to_iterable(k):
        init_lambda = [0.0]*k + [1.0]
        k_init_lambda.append((k, init_lambda))
    dd_stepsizes = v['stepsize_schedule']

    # other exp setup --------------------------------------
    exp_name = "{exp_index}_{time}_{game}".format(
        exp_index=exp_index,
        time=get_time_stamp(),
        game=game,
    )
    if ("ec2" in mode) and (len(exp_name) > 64):
        print("Should not use experiment name with length %d > 64.\nThe experiment name is %s.\n Exit now."%(len(exp_name),exp_name))
        sys.exit(1)

    adv_experiment = AdvSleeperExperiment(
            game,
            norm,
            fgsm_eps,
            [experiment],
            model_dir,
            save_rollouts,
            test_transfer,
            adversary_algo,
            adversary_algo_param_names,
            k_init_lambda=k_init_lambda,
            dual_descent_stepsizes=dd_stepsizes,
            num_ts=num_ts,
            seed=seed,
            obs_min=obs_min,
            obs_max=obs_max,
            threshold_perf=threshold_perf,
            threshold_n=threshold_n,
            N=N,
            across_rollout=across_rollout
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
        stub_method_call=adv_experiment.run(),
        exp_prefix=exp_prefix,
        exp_name=exp_name,
        seed=seed,
        n_parallel=1,
        mode=actual_mode,
        variant=v,
        use_gpu=use_gpu,
        plot=plot,
        sync_s3_pkl=False,
        sync_s3_log=False,
        sync_s3_h5=True,
        sync_log_on_termination=True,
        sync_all_data_node_to_s3=True,
        terminate_machine=terminate_machine,
        use_cloudpickle=False
    )
    if "test" in mode:
        sys.exit(0)

# make the current script read-only to avoid accidental changes after ec2 runs
if "local" not in mode:
    os.system("chmod 444 %s" % (__file__))

