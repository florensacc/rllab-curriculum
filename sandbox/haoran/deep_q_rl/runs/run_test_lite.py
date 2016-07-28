from __future__ import print_function
from __future__ import absolute_import
from rllab import config
from rllab.misc.instrument import VariantGenerator, variant
from rllab.misc.instrument import stub, run_experiment_lite
# import launcher
import sys
sys.path.append("sandbox/haoran/deep_q_rl/deep_q_rl")
from sandbox.haoran.deep_q_rl.deep_q_rl.launcher import Launcher
from rllab.misc.ext import AttrDict

stub(globals())


"""
Fix to counting scheme. Fix config...
"""

class VG(VariantGenerator):
    @variant
    def seed(self):
        return [311, 411, 511, 611, 711, 811, 911]

    @variant
    def bonus_coeff(self):
        return [0., 0.1, 0.01, 0.001]

    @variant
    def dim_key(self):
        return [64, 256]

    @variant
    def game(self):
        return ["montezuma_revenge", "freeway", "breakout", "frostbite"]

variants = VG().variants()

# config.KUBE_DEFAULT_NODE_SELECTOR = {
#     "aws/type": "m4.2xlarge",
# }
# config.KUBE_DEFAULT_RESOURCES = {
#     "requests": {
#         "cpu": 3.7,
#     },
#     "limits": {
#         "cpu": 3.7,
#     },
# }


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
        ROM = 'breakout.bin',
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
        DETERMINISTIC = True,
        CUDNN_DETERMINISTIC = False,
        USE_DOUBLE = True,
        CLIP_REWARD = True,
        USE_BONUS = True,
        AGENT_UNPICKLABLE_LIST = ["data_set","test_data_set"],
)
launcher = Launcher([], defaults, __doc__)
for v in variants:
    run_experiment_lite(
        stub_method_call=launcher.launch(),
        exp_prefix="hashing-test",
        seed=v["seed"],
        n_parallel=4,
        snapshot_mode="all",
        mode="local",
        # variant=v,
    )
