import joblib
import pickle
import random
import numpy as np

from gpr import Trajectory

file = "/tmp/params.pkl"

with open(file, "rb") as f:
    data = pickle.load(f)

env = data['env']

# while True:

import time

samples = np.transpose(data['samples'], axes=(1, 0, 2))
# sort trajs in decreasing order of return
samples = sorted(zip(samples, data['S']), key=lambda x: -x[1])
for (sample, S) in samples:
    print(S)
    env.reset_to(data['xinit'])
    env.render()
    S_check = 0
    # sample = random.choice(samples)
    for t, u in enumerate(sample):
        print(t)
        time.sleep(0.002)
        _, reward, _, _ = env.step(u)
        S_check += reward
        env.render()
    print("checked: ", S_check)

# for sample in data['samples']:
#         traj = Trajectory(env)
#         traj.solution = dict(xinit=data['xinit'], u=sample)
#         env.render()
#         env.view
#
# import ipdb; ipdb.set_trace()
