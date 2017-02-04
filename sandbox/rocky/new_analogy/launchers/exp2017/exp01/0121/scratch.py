from gpr.runner import TrajectoryRunner
from sandbox.rocky.new_analogy import fetch_utils
from sandbox.rocky.new_analogy.exp_utils import terminate_ec2_instances

# terminate_ec2_instances(
#     exp_name="fetch-relative-general-dagger-1-4",
#     filter_dict=dict(
#         hidden_sizes=[(1024, 1024, 1024),
#     )
# )


env = fetch_utils.fetch_env(horizon=1000, height=4)
path = fetch_utils.demo_path(env=env, animated=True)
trajectory = fetch_utils.path_to_traj(gpr_env=env.wrapped_env.gpr_env, path=path)

TrajectoryRunner([trajectory]).run()
