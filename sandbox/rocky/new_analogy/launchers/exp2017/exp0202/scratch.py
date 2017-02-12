# Try to evaluate wavenet policy

# register the file first
import itertools

from bin.tower_copter_policy import get_task_from_text
from sandbox.rocky.new_analogy import fetch_utils
from sandbox.rocky.new_analogy.th.policies.conv_analogy_policy_new import ConvAnalogyPolicy
from sandbox.rocky.s3 import resource_manager
import joblib
import numpy as np

# resource_manager.register_file("fetch_wavenet_test_v1.pkl", "/tmp/params.pkl")

# file_path = resource_manager.get_file("fetch_wavenet_test_v1.pkl")

# with open(file_path, "rb") as f:
# data = joblib.load("/tmp/params.pkl")#file_path)

# policy = data['policy']
# assert isinstance(policy, ConvAnalogyPolicy)


all_task_ids = list(map("".join, itertools.permutations("abcde", 2)))
np.random.RandomState(0).shuffle(all_task_ids)

task_id = all_task_ids[0]


n_configurations = 100  # 0
task_paths_resource = "fetch_analogy_paths/task_{}_trajs_{}.pkl".format(
    task_id, n_configurations)

task_paths_filename = resource_manager.get_file(task_paths_resource)
paths = np.asarray(joblib.load(task_paths_filename))

paths = [p for p in paths if len(p["rewards"]) == 500]

obs = np.asarray([p["observations"] for p in paths])

env = fetch_utils.fetch_env(horizon=500, height=5,
                            task_id=get_task_from_text(task_id))

np.random.seed(1)

policy = ConvAnalogyPolicy(
    env_spec=fetch_utils.discretized_env_spec(
        env.spec, fetch_utils.disc_intervals),
    rates=(1, 2, 4, 8, 16, 32, 64, 128, 256),
    residual_channels=256,
    filter_size=2,
)

policy.inform_task(task_id=task_id, env=env, paths=paths, obs=obs)

policy.fast_reset()
# policy.reset()

@profile
def run():
    global env
    env = fetch_utils.DiscretizedEnvWrapper(env, fetch_utils.disc_intervals)

    obs = env.reset()


    for _ in range(100):
        action, agent_info = policy.fast_get_action(obs)
        # action, agent_info = policy.get_action(obs)
        obs, reward, _, _ = env.step(action)
        print(reward)#, policy.distribution.entropy(
            # {k: np.expand_dims(v, 0) for k, v in agent_info.items()})[0])

run()
