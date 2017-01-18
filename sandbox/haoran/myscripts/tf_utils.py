import tensorflow as tf
from rllab import config

DTYPE = tf.float32

""" set up personalized tf session config """
config_args = dict(
    gpu_options=tf.GPUOptions(
        allow_growth=config.TF_GPU_ALLOW_GROWTH,
        per_process_gpu_memory_fraction=config.TF_GPU_MEM_FRAC,
    ),
    log_device_placement=config.TF_LOG_DEVICE_PLACEMENT,
)
if not config.TF_USE_GPU:
    config_args["device_count"] = {'GPU':0}
config = tf.ConfigProto(**config_args)

""" this should overwrite tf.Session() """
def create_session(**kwargs):
    if "config" not in kwargs:
        kwargs["config"] = config
    return tf.InteractiveSession(**kwargs)
