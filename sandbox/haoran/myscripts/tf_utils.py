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

"""
WARNING: don't use this code as it doesn't clip the actual step
"""
def adam_clipped_op(loss, var_list, lr, clip):
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    gvs = optimizer.compute_gradients(loss, var_list=var_list)
    capped_gvs = [(tf.clip_by_value(grad, -clip, clip), var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(capped_gvs)
    return optimizer, train_op
