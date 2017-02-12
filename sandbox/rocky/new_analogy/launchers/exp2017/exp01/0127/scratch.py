from rllab.misc import logger
from rllab.sampler.utils import rollout
from sandbox.rocky.new_analogy.exp_utils import run_local_docker, run_local
import joblib
import numpy as np


# Compute success rate of prescribed policy

# def run_task(vv):
#     from sandbox.rocky.new_analogy import fetch_utils
#     import numpy as np
#     # compute prescribed success rate
#
#     env = fetch_utils.fetch_env(horizon=2000, height=5)
#     policy = fetch_utils.fetch_prescribed_policy(env)
#
#     paths = fetch_utils.new_policy_paths(env=env, policy=policy, seeds=np.arange(100))
#
#     env.log_diagnostics(paths)
#     logger.dump_tabular()
#     joblib.dump(paths, "/tmp/fetch_5_boxes_prescribed_100_trajs.pkl", compress=3)
#     from sandbox.rocky.s3 import resource_manager
#     resource_manager.register_file("fetch_5_boxes_prescribed_100_trajs.pkl", "/tmp/fetch_5_boxes_prescribed_100_trajs.pkl")
#
#     import ipdb;
#     ipdb.set_trace()
#
# def run_task(vv):
#     from sandbox.rocky.new_analogy import fetch_utils
#     import numpy as np
#     # compute prescribed success rate
#
#     env = fetch_utils.fetch_env(horizon=2000, height=5)
#     policy = fetch_utils.fetch_discretized_prescribed_policy(env, fetch_utils.disc_intervals)
#
#     paths = fetch_utils.new_policy_paths(env=env, policy=policy, seeds=np.arange(100))
#
#     env.log_diagnostics(paths)
#     logger.dump_tabular()
#
#     name = "fetch_5_boxes_disc_prescribed_100_trajs"
#     joblib.dump(paths, "/tmp/{}.pkl".format(name), compress=3)
#     from sandbox.rocky.s3 import resource_manager
#     resource_manager.register_file("{}.pkl".format(name),
#                                    "/tmp/{}.pkl".format(name))
#
#     import ipdb;
#     ipdb.set_trace()




# name = "fetch_5_boxes_prescribed_100_trajs"
#
# file_name = resource_manager.get_file("{}.pkl".format(name))
#
# paths = joblib.load(file_name)
#
# env = fetch_utils.fetch_env(horizon=2000, height=5)
#
# paths = [p for p in paths if not env.wrapped_env._is_success(p)]
#
# for p in paths:
#     for x in p["env_infos"]["x"]:
#
#         fetch_utils.get_gpr_env(env).reset_to(x)#xinits[0])
#         env.render()

# xinits = np.asarray([p["env_infos"]["x"][0] for p in paths])
#
# import ipdb; ipdb.set_trace()
#
#
# fetch_utils.get_gpr_env(env).reset_to(xinits[0])
#
# while True:
#     env.render()

# import ipdb; ipdb.set_trace()

def run_task(vv):
    from sandbox.rocky.new_analogy import fetch_utils
    import numpy as np
    from sandbox.rocky.s3 import resource_manager
    import tensorflow as tf
    # compute prescribed success rate

    with tf.Session():

        env = fetch_utils.fetch_env(horizon=2000, height=5)

        policy = joblib.load(resource_manager.get_file("fetch_5_boxes_trpo_finetuned_policy.pkl"))['policy']

        policy = fetch_utils.DiscretizedFetchWrapperPolicy(
            policy,
            disc_intervals=fetch_utils.disc_intervals,
            expectation='log_mean'
        )

        # policy = fetch_utils.DeterministicPolicy(env_spec=env.spec, wrapped_policy=policy)

        rollout(env, policy, animated=True)

        paths = fetch_utils.new_policy_paths(env=env, policy=policy, seeds=np.arange(100))

        env.log_diagnostics(paths)
        logger.dump_tabular()

        name = "fetch_5_boxes_trpo_finetuned_mean_100_trajs"
        joblib.dump(paths, "/tmp/{}.pkl".format(name), compress=3)
        resource_manager.register_file("{}.pkl".format(name),
                                       "/tmp/{}.pkl".format(name))

        import ipdb;
        ipdb.set_trace()

# run_local_docker(
run_local(
    run_task,
    seed=0,
    n_parallel=0,
)
