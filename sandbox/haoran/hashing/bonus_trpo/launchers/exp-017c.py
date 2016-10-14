"""
Re-run exp-017
Extremely simplify the environment.
- short horizon (less get stuck)
- more hacky hash (has_key, is_skull_dead)
- terminate whenever the right door is opened
- frame_skip = 8
"""
# imports -----------------------------------------------------
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
from sandbox.haoran.hashing.bonus_trpo.envs.atari_env_info import semantic_actions

""" resetter """
# from sandbox.haoran.hashing.bonus_trpo.resetter.atari_count_resetter import AtariCountResetter
from sandbox.haoran.hashing.bonus_trpo.resetter.atari_save_load_resetter import AtariSaveLoadResetter

""" terminator """
from sandbox.haoran.hashing.bonus_trpo.terminators.montezuma_terminator import MontezumaTerminator

""" bonus """
from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.ale_hashing_bonus_evaluator import ALEHashingBonusEvaluator
from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.preprocessor.identity_preprocessor import IdentityPreprocessor
from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.hash.ale_hacky_hash_v3 import ALEHackyHashV3

""" others """
from sandbox.haoran.myscripts.myutilities import get_time_stamp
from sandbox.haoran.ec2_info import instance_info, subnet_info
from rllab import config
from rllab.misc.instrument import stub, run_experiment_lite
import sys,os
import copy

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant

# exp setup -----------------------------------------------------
exp_index = os.path.basename(__file__).split('.')[0] # exp_xxx
exp_prefix = "bonus-trpo-atari/" + exp_index
mode = "kube"
ec2_instance = "c4.8xlarge"
subnet = "us-west-1a"
config.DOCKER_IMAGE = "tsukuyomi2044/rllab3" # needs psutils

n_parallel = 1 # only for local exp
snapshot_mode = "last"
plot = False
use_gpu = False
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

# variant params ---------------------------------------
class VG(VariantGenerator):
    @variant
    def seed(self):
        return [0,100,200,300,400,500,600,700,800,900]
    @variant
    def bonus_coeff(self):
        return [1e-4]
    @variant
    def baseline_type_opt(self):
        return [
            # ["conv","cg"],
            ["nn_feature_linear",""],
        ]
    @variant
    def game(self):
        return ["montezuma_revenge"]
    @variant
    def resetter_type(self):
        return [""]
    @variant
    def task(self):
        return ["to_second_room"]
    @variant
    def cg_iters(self):
        return [10]
    @variant
    def subsample_factor(self):
        return [0.1,0.3]
    @variant
    def center_adv(self):
        return [True]
variants = VG().variants()


print("#Experiments: %d" % len(variants))
for v in variants:
    # non-variant params ------------------------------
    # algo
    use_parallel = True
    seed=v["seed"]
    if mode == "local_test":
        batch_size = 500
    else:
        batch_size = 50000
    max_path_length = 1500
    discount = 0.99
    n_itr = 1000
    step_size = 0.01
    clip_reward = True
    center_adv=v["center_adv"]
    policy_opt_args = dict(
        name="pi_opt",
        cg_iters=v["cg_iters"],
        reg_coeff=1e-3,
        subsample_factor=v["subsample_factor"],
        max_backtracks=15,
        backtrack_ratio=0.8,
        accept_violation=False,
        hvp_approach=None,
        num_slices=1, # reduces memory requirement
    )

    # env
    game=v["game"]
    frame_skip=8
    max_start_nullops = 0
    network_args = nips_dqn_args
    img_width=42
    img_height=42
    obs_type = "image"
    record_image=False
    record_rgb_image=False
    record_ram=True
    record_internal_state=False
    legal_semantic_actions = [
        "noop","fire",
        "up","right","left","down",
        "up-fire","right-fire","left-fire","down-fire",
    ] # disable diagonal movement
    legal_actions=[
        semantic_actions.index(s_action)
        for s_action in legal_semantic_actions
    ]

    # bonus
    bonus_coeff=v["bonus_coeff"]
    bonus_form="1/sqrt(n)"
    count_target="ram_states"
    retrieve_sample_size=10000

    # others
    resetter_type = v["resetter_type"]
    baseline_prediction_clip = 100
    task = v["task"]

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
    if resetter_type == "SL":
        # resetter = AtariSaveLoadResetter(
        #     restored_state_folder=None,
        #     avoid_life_lost=False,
        # )
        raise NotImplementedError
    else:
        resetter = None

    if task == "":
        terminator = None
    else:
        terminator = MontezumaTerminator(task=task)

    env = AtariEnv(
        game=game,
        seed=seed,
        max_start_nullops=max_start_nullops,
        img_width=img_width,
        img_height=img_height,
        obs_type=obs_type,
        record_ram=record_ram,
        record_image=record_image,
        record_rgb_image=record_rgb_image,
        record_internal_state=record_internal_state,
        resetter=resetter,
        terminator=terminator,
        frame_skip=frame_skip,
        legal_actions=legal_actions,
    )
    policy = CategoricalConvPolicy(
        env_spec=env.spec,
        name="policy",
        **network_args
    )

    # baseline
    baseline_type, baseline_opt = v["baseline_type_opt"]
    if baseline_type == "nn_feature_linear":
        baseline = ParallelNNFeatureLinearBaseline(
            env_spec=env.spec,
            policy=policy,
            nn_feature_power=1,
            t_power=3,
            prediction_clip=baseline_prediction_clip,
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
                batchsize=batch_size*10,
                normalize_inputs=True,
                normalize_outputs=True,
                **network_args_for_vf
            )
        )
    else:
        raise NotImplementedError

    # bonus
    if count_target == "images" or \
    (count_target == "observations" and obs_type == "image"):
        total_pixels=img_width * img_height
        state_preprocessor = SlicingPreprocessor(
            input_dim=total_pixels * n_last_screens,
            start=total_pixels * (n_last_screens - 1),
            stop=total_pixels * n_last_screens,
            step=1,
        )
    elif count_target == "ram_states":
        state_preprocessor = None
    else:
        raise NotImplementedError

    _hash = ALEHackyHashV3(
        item_dim=128,
        game=game,
        parallel=use_parallel,
    )
    bonus_evaluator = ALEHashingBonusEvaluator(
        log_prefix="",
        state_dim=128,
        state_preprocessor=None,
        hash=_hash,
        bonus_form=bonus_form,
        count_target=count_target,
        parallel=use_parallel,
        retrieve_sample_size=retrieve_sample_size,
    )

    algo = ParallelTRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        bonus_evaluator=bonus_evaluator,
        bonus_coeff=bonus_coeff,
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
        clip_reward=clip_reward,
        center_adv=center_adv,
    )

    if use_parallel:
        run_experiment_lite(
            algo.train(),
            exp_prefix=exp_prefix,
            exp_name=exp_name,
            seed=seed,
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
