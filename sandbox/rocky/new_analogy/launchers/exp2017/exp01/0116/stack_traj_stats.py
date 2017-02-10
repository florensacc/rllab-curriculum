import os
import pickle
import numpy as np

env_ = [None]
root_path = "/local_home/rocky/conopt-local-data/stack_ab_trajs_v2"
processed_root_path = "/local_home/rocky/conopt-local-data/stack_ab_trajs_v2_processed"

from gpr.envs.stack import Experiment

expr = Experiment(nboxes=2, horizon=500)
env = expr.make(task_id=[[1, "top", 0]])

model = env.world.model
site_names = model.site_names
geom0_idx = site_names.index('geom0')
geom1_idx = site_names.index('geom1')
dimq = model.dimq

cnt = 0
success_cnt = 0
final_rewards = []
successes = []
# print(len(os.listdir(processed_root_path)))


ranked = []
for idx, traj in enumerate(os.listdir(processed_root_path)):  # trajs

    if 'path' in traj:
        path = pickle.load(open(os.path.join(processed_root_path, traj), "rb"))
        site_xpos = path["observations"][:, 2 * dimq:]
        geom0_xpos = site_xpos[:, geom0_idx * 3:geom0_idx * 3 + 3]
        geom1_xpos = site_xpos[:, geom1_idx * 3:geom1_idx * 3 + 3]
        dist = np.linalg.norm(geom0_xpos - geom1_xpos, axis=1)
        cnt += 1
        final_rewards.append(path["rewards"][-1])
        success_t = np.where(dist < 0.06)[0]#[0]
        if len(success_t) > 0:
            successes.append(success_t[0])
            ranked.append((success_t[0], traj))
        if dist[-1] < 0.06:#np.min(dist) < 0.06:
            success_cnt += 1
        print(np.linalg.norm(path["observations"][0]))
        print("{}: Success rate: {}, min reward: {}, max reward: {}".format(idx, success_cnt / cnt,
                                                                            np.min(final_rewards),
                                                                            np.max(final_rewards)))
        print("Min success: {}, max: {}, avg: {}, median: {}".format(np.min(successes), np.max(successes),
              np.mean(successes),
              np.median(successes)))

# ranked = sorted(ranked, key=lambda x: x[0])
# for t, traj in ranked[:10]:
#     print(t, traj)



# import ipdb; ipdb.set_trace()
#
#
#
#
# def fast_set_state(self, state):
#     if env_[0] is None:
#         print("Creating env")
#         from gpr.env import Env
#         env_[0] = Env(**state['env'])
#     state['env'] = env_[0]
#     state['world'] = state['env'].world
#     self.__dict__.update(state)
#
#
# def process_traj(traj_file):
#     try:
#         from gpr import Trajectory
#         Trajectory.__setstate__ = fast_set_state
#
#         if "traj" not in traj_file:
#             return
#         traj_id = traj_file.split('_')[-1].split('.')[0]
#
#         processed_path = os.path.join(processed_root_path, "path_{0}.pkl".format(traj_id))
#         if not os.path.exists(processed_path):
#             print(traj_file)
#             full_path = os.path.join(root_path, traj_file)
#             with open(full_path, "rb") as f:
#                 traj = pickle.load(f)
#             process(traj, processed_path)
#     except Exception as e:
#         print(e)
#
#
# def process(traj, processed_path):
#     # get observations
#     observations = []
#     print("Getting obs")
#     for x in traj.solution["x"]:
#         obs = traj.env.world.observe(x)[0]
#         observations.append(obs)
#     print("Got obs")
#     path = dict(
#         observations=np.asarray(observations),
#         actions=np.asarray(traj.solution["u"]),
#         rewards=np.asarray(traj.solution["reward"]).flatten(),
#     )
#     with open(processed_path, "wb") as f:
#         pickle.dump(path, f)
#
#
# def run_task(*_):
#     # from gpr.env import Env
#     mkdir_p(processed_root_path)
#     traj_files = os.listdir(root_path)
#
#     # env_ = [None]
#
#     with multiprocessing.Pool() as pool:
#         pool.map(process_traj, traj_files)
#
#         # for traj_file in traj_files:
#
#
# run_task()
# # run_experiment_lite(
# #     run_task,
# #     use_cloudpickle=True,
# #     mode="local_docker",
# #     # exp_prefix="stack-pi2-3",
# #     # variant=dict(worker_id=worker_id),
# #     # terminate_machine=True,
# #     docker_args=" -v /home/rocky/conopt-shared-data:/shared-data",
# #     env=dict(PYTHONPATH="/root/code/rllab:/root/code/rllab/gpr_package"),
# #     # sync_log_on_termination=False,
# #     # periodic_sync=False,
# #     # sync_all_data_node_to_s3=False,
# # )
