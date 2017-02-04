# We know that pure BC is enough to get very good performance on stacking two blocks
# So let's start from here
from bin.tower_copter_policy import get_task_from_text
from sandbox.rocky.new_analogy import fetch_utils


# have a bunch of these envs?
# how to do vectorized sampling?
#
env = fetch_utils.fetch_env(height=5, task_id=get_task_from_text("cd"))
# policy = fetch_utils.fetch_discretized_prescribed_policy(env, fetch_utils.disc_intervals)
policy = fetch_utils.fetch_prescribed_policy(env)

fetch_utils.demo_path(env=env, policy=policy, animated=True)
# fetch_utils.new_policy_paths()

