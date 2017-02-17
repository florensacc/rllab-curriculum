"""
Test paralell neural network baselines
"""
from sandbox.haoran.parallel_trpo.trpo import ParallelTRPO
from sandbox.haoran.parallel_trpo.conjugate_gradient_optimizer import ParallelConjugateGradientOptimizer
from sandbox.adam.parallel.gaussian_conv_baseline import ParallelGaussianConvBaseline
from sandbox.adam.modified_sampler.batch_sampler import BatchSampler
from sandbox.haoran.myscripts.retrainer import Retrainer
from sandbox.haoran.mddpg.envs.meta_env import MetaEnvTheano
from rllab.algos.trpo import TRPO
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.haoran.mddpg.policies.stochastic_policy_theano import \
    StochasticNNPolicy, PolicyCopier
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.normalized_env import normalize

from sandbox.haoran.myscripts.myutilities import \
    get_time_stamp, MultiTasker
from sandbox.haoran.ec2_info import instance_info, subnet_info
from sandbox.haoran.myscripts.envs import EnvChooser
from rllab import config
from rllab.misc.instrument import stub, run_experiment_lite

import lasagne
import sys,os
import copy

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant

exp_index = os.path.basename(__file__).split('.')[0] # exp_xxx
exp_prefix = "tuomas/vddpg/" + exp_index

mode = "local"
ec2_instance = "c4.8xlarge"
subnet = "us-west-1c"
config.DOCKER_IMAGE = "tsukuyomi2044/rllab3" # needs psutils
config.AWS_IMAGE_ID = "ami-85d181e5" # with docker already pulled

n_parallel = 4 # only for local exp
snapshot_mode = "gap"
snapshot_gap = 10
plot = False


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [0,100,200]

    @variant
    def env_name(self):
        return ["tuomas_ant"]

    @variant
    def exp_info(self):
        return [
            # this Ant can move in many directions fast
            dict(
                exp_prefix="tuomas/vddpg/exp-000b",
                exp_name="exp-000b_20170213_150257_941463_tuomas_ant",
                snapshot_file="itr_499.pkl",
                env_name="tuomas_ant",
                seed=0
            ),
        ]
    @variant
    def random_init_state(self):
        return [False]

    @variant
    def direction(self):
        return [(1., 0.)]


variants = VG().variants()
print("#Experiments: %d" % len(variants))
for v in variants:
    # params ---------------------------------------
    step_size = 0.01
    max_path_length = 500
    discount = 0.99

    if mode in ["local_test", "local_docker_test"]:
        batch_size = 1000
        n_itr = 5
    else:
        batch_size = 50000
        n_itr = 1000

    policy_opt_args = dict(
        cg_iters=10,
        subsample_factor=0.2,
        name="pi_opt"
    ) # worth tuning to see the speed up by parallelism

    if sys.platform == "darwin":
        set_cpu_affinity = False
        cpu_assignments = None
        serial_compile = False
    else:
        set_cpu_affinity = True
        cpu_assignments = None
        serial_compile = True

    env_name = v["env_name"]
    if env_name == "tuomas_ant":
        env_kwargs = {
            "reward_type": "velocity",
            "direction": v["direction"],
            "random_init_state": v["random_init_state"],
        }
    else:
        env_kwargs = {}
    # ----------------------------------------------------------------------
    exp_name = "{exp_index}_{time}_{env_name}".format(
        exp_index=exp_index,
        time=get_time_stamp(),
        env_name=env_name
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

    # construct objects ----------------------------------
    env_chooser = EnvChooser()
    base_env = normalize(
        env_chooser.choose_env(env_name,**env_kwargs),
        clip=True,
    )
    transform_policy = StochasticNNPolicy.load_from_file(
        file_name="/tmp/transform_policy.pkl"
    )
    meta_env = MetaEnvTheano(
        env=base_env,
        transform_policy=transform_policy,
    )
    env = normalize(meta_env)

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(32, 32),
        init_std=meta_env.get_default_sigma_after_normalization(),
    )
    baseline = ParallelGaussianConvBaseline(
        env_spec=env.spec,
        regressor_args=dict(
            hidden_sizes=(32,32),
            conv_filters=[],
            conv_filter_sizes=[],
            conv_strides=[],
            conv_pads=[],
            batchsize=batch_size,
            normalize_inputs=True,
            normalize_outputs=True, #???
            optimizer=ParallelConjugateGradientOptimizer(
                subsample_factor=1.0,
                cg_iters=10,
                name="vf_opt"
            )
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
        plot=plot,
        step_size=step_size,
        store_paths=False,
        n_parallel=n_parallel,
        optimizer_args=policy_opt_args,
        set_cpu_affinity=set_cpu_affinity,
        cpu_assignments=cpu_assignments,
        serial_compile=serial_compile,
    )

    retrainer = Retrainer(
        exp_prefix=v["exp_info"]["exp_prefix"],
        exp_name=v["exp_info"]["exp_name"],
        snapshot_file=v["exp_info"]["snapshot_file"],
        configure_script="",
    )
    copier = PolicyCopier(
        retrainer=retrainer,
        file_name="/tmp/transform_policy.pkl"
    )
    multitakser = MultiTasker(
        [(copier.run, {}), (algo.train, {})]
    )


    # run --------------------------------------------------
    batch_tasks = []
    batch_tasks.append(
        dict(
            stub_method_call=copier.run(),
            exp_name=exp_name,
        )
    )
    batch_tasks.append(
        dict(
            stub_method_call=algo.train(),
            # stub_method_call=multitakser.run(),
            exp_name=exp_name,
            seed=v["seed"],
            snapshot_mode=snapshot_mode,
            snapshot_gap=snapshot_gap,
            variant=v,
            plot=plot,
            n_parallel=n_parallel,
        )
    )
    run_experiment_lite(
        batch_tasks=batch_tasks,
        exp_prefix=exp_prefix,
        mode=actual_mode,
        sync_s3_pkl=True,
        sync_s3_log=True,
        sync_s3_png=True,
        sync_log_on_termination=True,
        sync_all_data_node_to_s3=True,
        terminate_machine=("test" not in mode),
        use_cloudpickle=False,
    )
    if "test" in mode:
        sys.exit(0)

if ("local" not in mode) and ("test" not in mode):
    os.system("chmod 444 %s"%(__file__))
