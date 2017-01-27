import pickle
from multiprocessing import cpu_count

from rllab.sampler import parallel_sampler
from sandbox.rocky.s3 import resource_manager

parallel_sampler.initialize(n_parallel=cpu_count())
from sandbox.rocky.new_analogy import fetch_utils
import numpy as np

for n_trajs in [100, 1000, 10000]:
    env = fetch_utils.fetch_env(horizon=300)
    policy = fetch_utils.fetch_prescribed_policy(env.wrapped_env.gpr_env)
    paths = fetch_utils.demo_paths(env=env, seeds=np.arange(n_trajs), noise_levels=[0., 0.001, 0.01, 0.1])

    file_name = resource_manager.tmp_file_name(file_ext="pkl")
    with open(file_name, "wb") as f:
        pickle.dump(paths, f)
    print("Success rate: ", np.mean([p["rewards"][-1] > 5 for p in paths]))
    resource_manager.register_file(resource_name="fetch_relative/{}_trajs.pkl".format(n_trajs), file_name=file_name)
