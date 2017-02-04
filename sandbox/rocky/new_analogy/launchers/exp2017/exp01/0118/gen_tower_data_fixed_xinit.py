import pickle
import tempfile

from gym.spaces import prng

from rllab.misc.instrument import run_experiment_lite
from rllab.sampler import parallel_sampler
from sandbox.rocky.new_analogy.scripts.tower.gpr_policy_wrapper import GprPolicyWrapper
from sandbox.rocky.tf.envs.base import TfEnv

from sandbox.rocky.s3.resource_manager import resource_manager
import numpy as np


def gen_data(*_):
    from sandbox.rocky.new_analogy.envs.gpr_env import GprEnv
    from gpr_package.bin import tower_fetch_policy as tower
    np.random.seed(0)
    task_id = tower.get_task_from_text("ab")

    horizon = 1000

    stds = np.asarray([0.1, 0.1, 0.1, 0., 0., 0., 1., 1.])

    env = TfEnv(GprEnv("fetch.sim_fetch", task_id=task_id, experiment_args=dict(nboxes=2, horizon=horizon)))

    gpr_env = env.wrapped_env.gpr_env
    gpr_env.seed(0)
    prng.seed(0)

    xinit = gpr_env.world.sample_xinit()

    env = TfEnv(GprEnv("fetch.sim_fetch", task_id=task_id, experiment_args=dict(nboxes=2, horizon=horizon),
                       xinits=[xinit]))

    for n_trajs in [100, 1000, 10000]:
        for p_rand_action in [x * 0.1 for x in range(11)]:

            for std_level in [1.]:
                policy = GprPolicyWrapper(tower.FetchPolicy(task_id), stds=std_level * stds,
                                          p_rand_action=p_rand_action)
                parallel_sampler.populate_task(env=env, policy=policy)

                paths = parallel_sampler.sample_paths(None, max_samples=n_trajs * horizon,
                                                      max_path_length=horizon)
                print("P rand action:", p_rand_action, "Std level: ", std_level,
                      "Success rate: ", np.mean(np.asarray([p["rewards"][-1] for p in paths]) > 4))
                f_name = tempfile.NamedTemporaryFile().name + ".pkl"
                with open(f_name, "wb") as f:
                    pickle.dump(paths, file=f, protocol=pickle.HIGHEST_PROTOCOL)
                resource_manager.register_file("tower_fetch_ab_fixed_xinit/n_trajs_{0}_p_rand_{1}_std_level_{2}".format(
                    n_trajs, p_rand_action, std_level), f_name)


run_experiment_lite(
    gen_data,
    use_cloudpickle=True,
    mode="local_docker",
    n_parallel=32,
    seed=0,
    env=dict(CUDA_VISIBLE_DEVICES="0", PYTHONPATH="/root/code/rllab:/root/code/rllab/gpr_package"),
    docker_image="quay.io/openai/rocky-rllab3-gpr-gpu-pascal:20170114",
    docker_args=" -v /home/rocky/conopt-shared-data:/shared-data",
)
