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



# import matplotlib.pyplot as plt
# import networkx as nx
# import tensorflow as tf
# class TfGraphPlotter(object):
#     def children(self, op):
#       return set(op for out in op.outputs for op in out.consumers())
#
#     def get_graph(self):
#       """Creates dictionary {node: {child1, child2, ..},..} for current
#       TensorFlow graph. Result is compatible with networkx/toposort"""
#       print('get_graph')
#       ops = tf.get_default_graph().get_operations()
#       return {op: self.children(op) for op in ops}
#
#     def plot_graph(self, G):
#         '''Plot a DAG using NetworkX'''
#         def mapping(node):
#             return node.name
#         G = nx.DiGraph(G)
#         nx.relabel_nodes(G, mapping, copy=False)
#         nx.draw(G, cmap = plt.get_cmap('jet'), with_labels = True)
#         plt.show()
#
#     def show(self):
#         self.plot_graph(self.get_graph())
#         input("Press enter to continue...")
