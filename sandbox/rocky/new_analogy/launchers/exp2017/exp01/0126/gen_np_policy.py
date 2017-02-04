import joblib

from rllab.misc.console import mkdir_p
from sandbox.rocky.s3 import resource_manager
import os.path
import tensorflow as tf
import numpy as np
import pickle


file_name = resource_manager.get_file("fetch_5_blocks_dagger.pkl")

with tf.Session() as sess:
    data = joblib.load(file_name)

    policy = data['policy']

    intervals = policy.disc_intervals

    var_names = [x.name for x in policy.get_params()]
    var_vals = sess.run(policy.get_params())

    all_params = dict(zip(var_names, var_vals), intervals=intervals)

    W0 = all_params["policy/regressor/hidden_0/W:0"]
    b0 = all_params["policy/regressor/hidden_0/b:0"]
    W1 = all_params["policy/regressor/hidden_1/W:0"]
    b1 = all_params["policy/regressor/hidden_1/b:0"]
    W2 = all_params["policy/regressor/output/W:0"]
    b2 = all_params["policy/regressor/output/b:0"]
    intervals = all_params["intervals"]
    intervals = list(map(np.asarray, intervals))
    n_bins = list(map(len, intervals))

    with open("policy_data.pkl", "wb") as f:
        pickle.dump(all_params, f)

    # def get_action(ob):
    #     h0 = np.tanh(ob.dot(W0) + b0)
    #     h1 = np.tanh(h0.dot(W1) + b1)
    #     out = h1.dot(W2) + b2
    #
    #     a0 = intervals[0][np.argmax(out[:n_bins[0]])]
    #     a1 = intervals[1][np.argmax(out[n_bins[0]:n_bins[0]+n_bins[1]])]
    #     a2 = intervals[2][np.argmax(out[n_bins[0]+n_bins[1]:n_bins[0]+n_bins[1]+n_bins[2]])]
    #     a3 = [-1,1][np.argmax(out[-2:])]
    #
    #     # convert back to relative action space
    #     action = np.asarray([a0, a1, a2, 0, 0, 0, a3, a3])
    #     return action
    #
    # env = data['env']
    #
    # ob = env.reset()
    # while True:
    #     ob, _, _, _ = env.step(get_action(ob))
    #     env.render()


#     get_action(data['env'].reset())
#
# import ipdb; ipdb.set_trace()



# def get_action():





