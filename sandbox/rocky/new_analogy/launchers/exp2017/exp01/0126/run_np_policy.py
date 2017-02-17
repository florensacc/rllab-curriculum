from gym.spaces import prng

from gpr.envs import fetch_bc
import numpy as np
import pickle

with open("policy_data.pkl", "rb") as f:
    all_params = pickle.load(f)

W0 = all_params["policy/regressor/hidden_0/W:0"]
b0 = all_params["policy/regressor/hidden_0/b:0"]
W1 = all_params["policy/regressor/hidden_1/W:0"]
b1 = all_params["policy/regressor/hidden_1/b:0"]
W2 = all_params["policy/regressor/output/W:0"]
b2 = all_params["policy/regressor/output/b:0"]
intervals = all_params["intervals"]
intervals = list(map(np.asarray, intervals))
n_bins = list(map(len, intervals))


def get_action(ob):
    h0 = np.tanh(ob.dot(W0) + b0)
    h1 = np.tanh(h0.dot(W1) + b1)
    out = h1.dot(W2) + b2

    a0 = intervals[0][np.argmax(out[:n_bins[0]])]
    a1 = intervals[1][np.argmax(out[n_bins[0]:n_bins[0] + n_bins[1]])]
    a2 = intervals[2][np.argmax(out[n_bins[0] + n_bins[1]:n_bins[0] + n_bins[1] + n_bins[2]])]
    a3 = [-1, 1][np.argmax(out[-2:])]

    # convert back to relative action space
    action = np.asarray([a0, a1, a2, 0, 0, 0, a3, a3])
    return action


if __name__ == "__main__":

    # tweak the seed to change initial configuration. for some configuration it does not succeed
    seed = 600
    expr = fetch_bc.Experiment(horizon=2000)
    env = expr.make(height=5)

    prng.seed(seed)
    env.seed(seed)
    ob = env.reset()
    while True:
        ob, _, _, _ = env.step(get_action(ob))
        env.render()
