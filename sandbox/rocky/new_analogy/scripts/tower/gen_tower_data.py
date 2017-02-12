import pickle
import tempfile

from rllab.misc.instrument import run_experiment_lite
from rllab.sampler import parallel_sampler
from sandbox.rocky.new_analogy.scripts.tower.gpr_policy_wrapper import GprPolicyWrapper
from sandbox.rocky.tf.envs.base import TfEnv

from sandbox.rocky.s3.resource_manager import resource_manager
import numpy as np


def gen_data(*_):
    from sandbox.rocky.new_analogy.envs.gpr_env import GprEnv
    from gpr_package.bin import tower_copter_policy as tower
    from sandbox.rocky.new_analogy.scripts.tower.crippled_policy import CrippledPolicy
    np.random.seed(0)
    task_id = tower.get_task_from_text("ab")

    env = TfEnv(GprEnv("tower", task_id=task_id, experiment_args=dict(nboxes=2, horizon=1000)))
    policy = GprPolicyWrapper(CrippledPolicy(tower.CopterPolicy(task_id)))
    parallel_sampler.populate_task(env=env, policy=policy)

    max_path_length = 1000

    for n_trajs in [100, 1000, 10000]:
        paths = parallel_sampler.sample_paths(None, max_samples=n_trajs * max_path_length,
                                              max_path_length=max_path_length)
        print("Success rate: ", np.mean(np.asarray([p["rewards"][-1] for p in paths]) > 4))
        f_name = tempfile.NamedTemporaryFile().name + ".pkl"
        with open(f_name, "wb") as f:
            pickle.dump(paths, file=f, protocol=pickle.HIGHEST_PROTOCOL)
        # np.savez_compressed(f_name, np.asarray(paths))
        resource_manager.register_file("tower_copter_paths_ab_crippled_{0}".format(n_trajs), f_name)#, compress=True)
        print("Done!")


# if __name__ == "__main__":

run_experiment_lite(
    gen_data,
    use_cloudpickle=True,
    mode="local_docker",
    n_parallel=32,
    seed=0,
    env=dict(CUDA_VISIBLE_DEVICES="0", PYTHONPATH="/root/code/rllab:/root/code/rllab/gpr_package"),
    docker_image="quay.io/openai/rocky-rllab3-gpr-gpu-pascal:20170111",
    docker_args=" -v /home/rocky/conopt-shared-data:/shared-data",
)
