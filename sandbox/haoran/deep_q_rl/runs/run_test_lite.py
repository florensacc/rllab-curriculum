from __future__ import print_function
from __future__ import absolute_import
from rllab import config
from rllab.misc.instrument import stub, run_experiment_lite
# import launcher
import sys,os
sys.path.append("sandbox/haoran/deep_q_rl/deep_q_rl")
from sandbox.haoran.deep_q_rl.deep_q_rl.launcher import Launcher
from sandbox.haoran.hashing.hash.sim_hash import SimHash
from sandbox.haoran.hashing.bonus_evaluators.ale_hashing_bonus_evaluator import ALEHashingBonusEvaluator
from sandbox.haoran.hashing.preprocessor.image_vectorize_preprocessor import ImageVectorizePreprocessor
from rllab.misc.ext import AttrDict

stub(globals())
os.path.join(config.PROJECT_PATH)


# define running mode specific params -----------------------------------
exp_prefix = os.path.basename(__file__).split('.')[0] # exp_xxx
mode = "local_test"
snapshot_mode = "all"
plot = True

# different training params ------------------------------------------
from rllab.misc.instrument import VariantGenerator, variant
class VG(VariantGenerator):
    @variant
    def seed(self):
        return [1,101]

    @variant
    def bonus_coeff(self):
        return [0., 1.]

    @variant
    def dim_key(self):
        return [64, 128]

    @variant
    def game(self):
        return ["breakout", "freeway"]

variants = VG().variants()

# mode specific settings -----------------------------------
if mode == "ec2_cpu":
    config.AWS_INSTANCE_TYPE = "m4.large"
    config.AWS_SPOT_PRICE = '0.1'
    plot = False
    raise NotImplementedError
elif mode == "ec2_gpu":
    config.AWS_INSTANCE_TYPE = "g2.2xlarge"
    config.AWS_SPOT_PRICE = '1.0'
    plot = False


# check for uncommitted changes ------------------------------
import git
repo = git.Repo('.')
if repo.is_dirty():
    answer = ''
    while answer not in ['y','Y','n','N']:
        answer = raw_input("The repository has uncommitted changes. Do you want to continue? (y/n)")
    if answer in ['n','N']:
        sys.exit(1)

for v in variants:
    defaults = AttrDict(
        # ----------------------
        # Experiment Parameters
        # ----------------------
        STEPS_PER_EPOCH = 300,
        EPOCHS = 2,
        STEPS_PER_TEST = 300,
        EXPERIMENT_DIRECTORY = None, # use default, see launcher.py
        EXPERIMENT_PREFIX = "data/local/deep_q_rl/",

        # ----------------------
        # ALE Parameters
        # ----------------------
        BASE_ROM_PATH = "sandbox/haoran/deep_q_rl/roms/",
        ROM = '%s.bin'%(v["game"]),
        FRAME_SKIP = 4,
        REPEAT_ACTION_PROBABILITY = 0,

        # ----------------------
        # Agent/Network parameters:
        # ----------------------
        UPDATE_RULE = 'rmsprop',
        BATCH_ACCUMULATOR = 'mean',
        LEARNING_RATE = .0002,
        DISCOUNT = .95,
        RMS_DECAY = .99, # (Rho)
        RMS_EPSILON = 1e-6,
        MOMENTUM = 0,
        CLIP_DELTA = 0,
        EPSILON_START = 1.0,
        EPSILON_MIN = .1,
        EPSILON_DECAY = 1000000,
        PHI_LENGTH = 4,
        UPDATE_FREQUENCY = 1,
        REPLAY_MEMORY_SIZE = 1000000,
        BATCH_SIZE = 32,
        NETWORK_TYPE = "nips_dnn",
        FREEZE_INTERVAL = 1,
        REPLAY_START_SIZE = 100,
        RESIZE_METHOD = 'crop',
        RESIZED_WIDTH = 84,
        RESIZED_HEIGHT = 84,
        DEATH_ENDS_EPISODE = 'false',
        MAX_START_NULLOPS = 0,
        DETERMINISTIC = False,
        CUDNN_DETERMINISTIC = False,
        USE_DOUBLE = True,
        CLIP_REWARD = True,
        AGENT_UNPICKLABLE_LIST = ["data_set","test_data_set"],
        SEED = v["seed"],
        DISPLAY_SCREEN = plot,
    )
    launcher = Launcher([], defaults, __doc__) # first argument must be empty

    img_preprocessor = ImageVectorizePreprocessor(
        n_chanllel=defaults.get("PHI_LENGTH"),
        width=defaults.get("RESIZED_WIDTH"),
        height=defaults.get("RESIZED_HEIGHT"),
    )
    hash_list = [
        SimHash(
            item_dim=img_preprocessor.get_output_dim(), # get around stub
            dim_key=v["dim_key"],
            bucket_sizes=None,
        )
    ]
    bonus_evaluator = ALEHashingBonusEvaluator(
        state_dim=img_preprocessor.get_output_dim(),
        img_preprocessor=img_preprocessor,
        num_actions=launcher.get_num_actions(),
        hash_list=hash_list,
        count_mode="s",
        bonus_mode="s_next",
        bonus_coeff=v["bonus_coeff"],
        state_bonus_mode="1/n_s",
        state_action_bonus_mode="log(n_s)/n_sa",
    )
    launcher.set_agent_attr("bonus_evaluator",bonus_evaluator) #!!!

    # define the exp_name (log folder name)
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

    if "local" in mode:
        actual_mode = "local"
    elif "ec2" in mode:
        actual_mode = "ec2"
    else:
        raise NotImplementedError

    # official run --------------------------------------------------
    run_experiment_lite(
        stub_method_call=launcher.launch(),
        exp_prefix=exp_prefix,
        exp_name=exp_name,
        seed=v["seed"],
        n_parallel=1, # we actually don't use parallel_sampler here
        snapshot_mode=snapshot_mode,
        mode=actual_mode,
        variant=v,
        terminate_machine=True,
    )
    if "test" in mode:
        sys.exit(0)

# logging -------------------------------------------------------------
# record the experiment names to a file
# also record the branch name and commit number
logs = []
logs += ["branch: %s" %(repo.active_branch.name)]
logs += ["commit SHA: %s"%(repo.head.object.hexsha)]
logs += exp_names

cur_script_name = __file__
log_file_name = cur_script_name.split('.py')[0] + '.log'
with open(log_file_name,'w') as f:
    for message in logs:
        f.write(message + "\n")

# make the current script read-only to avoid accidental changes after ec2 runs
if "local" not in mode:
    os.system("chmod 444 %s"%(__file__))
