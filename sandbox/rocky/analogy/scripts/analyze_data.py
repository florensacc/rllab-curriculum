import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from sandbox.rocky.analogy.envs.conopt_particle_env import ConoptParticleEnv
from rllab import config

data = np.load(os.path.join(config.PROJECT_PATH, "data/conopt-trajs/conopt-particle-2-30-full-state-v5/5050.npz"))
paths = data["demo_paths"]
dists = [np.sqrt(np.sum(np.square(p["observations"][0][4:].reshape(3,3)[0][:2] - p["observations"][0][4:].reshape(3,3)[1][:2]))) for p in paths]

print(np.histogram(dists))