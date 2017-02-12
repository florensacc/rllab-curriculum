import itertools

import joblib
import pyprind
import numpy as np

from rllab.misc import logger
from rllab.misc.ext import AttrDict
from sandbox.rocky.s3 import resource_manager
import tensorflow as tf
import keras
import keras.backend as K


# import torch
# from torch.autograd import Variable
# from torch import optim

# from sandbox.rocky.th import tensor_utils
# import torch.nn.functional as F
from sandbox.rocky.tf.misc import tensor_utils

"""
Try without dagger, new obs space
"""


class AnalogyDataset(object):
    def __init__(self):
        self.paths = dict()
        self.envs = dict()
        self.all_paths = []
        self.obs = dict()
        self.actions = dict()
        self.flat_obs = dict()
        self.flat_actions = dict()
        self.obs_dim = None
        self.action_dim = None

    @property
    def n_tasks(self):
        return len(self.paths)

    def add_paths(self, task_id, env, paths):
        for p in paths:
            p["task_id"] = task_id
        self.paths[task_id] = paths
        self.all_paths.extend(paths)
        self.envs[task_id] = env
        self.obs[task_id] = np.asarray([p["observations"] for p in paths])
        self.actions[task_id] = np.asarray([p["actions"] for p in paths])
        self.obs_dim = obs_dim = self.obs[task_id].shape[-1]
        self.action_dim = action_dim = self.actions[task_id].shape[-1]
        self.flat_obs[task_id] = self.obs[task_id].reshape((-1, obs_dim))
        self.flat_actions[task_id] = self.actions[task_id].reshape((-1, action_dim))

    def sample_bc_batch(self, demo_batch_size, per_demo_batch_size, per_demo_full_traj=False):
        # Sample a bunch of demonstration paths. Then, for each of them,
        # sample a bunch of observation / action pairs for behavior cloning training
        # (unless per_demo_batch_size=True, in which case, sample entire trajectories)
        self.all_paths = np.asarray(self.all_paths)
        demo_paths = np.random.choice(self.all_paths, size=demo_batch_size, replace=True)
        # make sure they are aligned
        demo_task_ids = np.asarray([p["task_id"] for p in demo_paths])
        uniq_task_ids = set(demo_task_ids)

        obs_dim = self.obs_dim
        disc_action_dim = self.action_dim

        if per_demo_full_traj:
            assert per_demo_batch_size == list(self.obs.values())[0].shape[1]

        batch_obs = np.zeros((demo_batch_size, per_demo_batch_size, obs_dim), dtype=np.float32)
        batch_actions = np.zeros((demo_batch_size, per_demo_batch_size, disc_action_dim))

        for task_id in uniq_task_ids:
            demo_mask = demo_task_ids == task_id
            demo_cnt = int(np.sum(demo_mask))
            if per_demo_full_traj:
                batch_ids = np.random.choice(
                    len(self.obs[task_id]),
                    size=demo_cnt,
                    replace=True
                )
                batch_obs[demo_mask] = self.obs[task_id][batch_ids]
                batch_actions[demo_mask] = self.actions[task_id][batch_ids]
            else:
                batch_ids = np.random.choice(
                    len(self.flat_obs[task_id]),
                    size=demo_cnt * per_demo_batch_size,
                    replace=True
                )
                batch_obs[demo_mask] = self.flat_obs[task_id][batch_ids].reshape(
                    (demo_cnt, per_demo_batch_size, obs_dim)
                )
                batch_actions[demo_mask] = self.flat_actions[task_id][batch_ids].reshape(
                    (demo_cnt, per_demo_batch_size, disc_action_dim)
                )

        return demo_paths, batch_obs, batch_actions


def FetchAnalogyTrainer(
        template_env,
        policy,
        demo_batch_size,  # select this number of demonstrations
        per_demo_batch_size,
        per_demo_full_traj=False,  # whether to sample a whole trajectory for each demonstration
        n_updates_per_epoch=10000,
        n_configurations=100,
        horizon=500,
        n_boxes=5,
        n_eval_paths_per_task=10,
        optimizer=None,
        evaluate_policy=True,
        n_train_tasks=16,
        n_test_tasks=4):
    from bin.tower_copter_policy import get_task_from_text
    from sandbox.rocky.new_analogy import fetch_utils

    if per_demo_full_traj:
        assert per_demo_batch_size == horizon

    env_spec = fetch_utils.discretized_env_spec(
        template_env.spec,
        disc_intervals=fetch_utils.disc_intervals
    )
    obs_dim = env_spec.observation_space.flat_dim
    disc_action_dim = env_spec.action_space.flat_dim

    def load_task_paths(task_id):
        task_id_arr = get_task_from_text(task_id)
        logger.log("Loading paths for task {}".format(task_id))
        task_paths_resource = "fetch_analogy_paths/task_{}_trajs_{}.pkl".format(task_id, n_configurations)
        task_paths_filename = resource_manager.get_file(task_paths_resource)
        env = fetch_utils.fetch_env(horizon=horizon, height=n_boxes, task_id=task_id_arr)
        paths = np.asarray(joblib.load(task_paths_filename))

        for p in paths:
            # recomputing stages
            p["env_infos"]["stage"] = fetch_utils.compute_stage(env, p["env_infos"]["site_xpos"])
            # map to discrete-valued actions
            disc_actions = []
            for disc_idx in range(3):
                cur_actions = p["actions"][:, disc_idx]
                bins = np.asarray(fetch_utils.disc_intervals[disc_idx])
                disc_actions.append(np.argmin(np.abs(cur_actions[:, None] - bins[None, :]), axis=1))
            disc_actions.append(np.cast['uint8'](p["actions"][:, -1] == 1))
            flat_actions = env_spec.action_space.flatten_n(np.asarray(disc_actions).T)
            p["actions"] = flat_actions

        is_success = lambda p: env.wrapped_env._is_success(p) and len(p["rewards"]) == horizon
        # filter out failed paths
        logger.log("Success rate for task {}: {}".format(task_id, np.mean(list(map(is_success, paths)))))
        paths = np.asarray(list(filter(is_success, paths)))

        logger.log("Loaded")
        return env, paths

    def train():
        import numpy as np

        all_task_ids = list(map("".join, itertools.permutations("abcde", 2)))
        np.random.RandomState(0).shuffle(all_task_ids)

        train_task_ids = all_task_ids[:n_train_tasks]
        test_task_ids = all_task_ids[n_train_tasks:n_train_tasks + n_test_tasks]

        xinits_file_name = resource_manager.get_file("fetch_1000_xinits_{}_boxes.pkl".format(n_boxes))
        xinits = joblib.load(xinits_file_name)

        train_dataset = AnalogyDataset()
        test_dataset = AnalogyDataset()

        logger.log("Generating initial demonstrations")
        for task_id in train_task_ids:
            train_env, train_paths = load_task_paths(task_id)
            train_dataset.add_paths(task_id=task_id, env=train_env, paths=train_paths)

        if evaluate_policy:
            for task_id in test_task_ids:
                test_env, test_paths = load_task_paths(task_id)
                test_dataset.add_paths(task_id=task_id, env=test_env, paths=test_paths)

        logger.log("Generated")

        eval_policy = fetch_utils.DeterministicPolicy(
            env_spec=env_spec,
            wrapped_policy=fetch_utils.DiscretizedFetchWrapperPolicy(
                wrapped_policy=policy,
                disc_intervals=fetch_utils.disc_intervals
            )
        )

        obs_var = env_spec.observation_space.new_tensor_variable(name="obs", extra_dims=2)
        action_var = env_spec.action_space.new_tensor_variable(name="action", extra_dims=2)
        demo_vars = policy.new_demo_vars()

        def init_opt():
            dist_info_vars = policy.dist_info_sym(obs_var=obs_var, demo_vars=demo_vars)
            reshaped_action_var = tf.reshape(action_var, (-1, disc_action_dim))
            logli_var = policy.distribution.log_likelihood_sym(reshaped_action_var, dist_info_vars)
            loss_var = -tf.reduce_mean(logli_var)
            if optimizer is None:
                new_optimizer = keras.optimizers.Adamax()
            else:
                new_optimizer = optimizer
            updates = new_optimizer.get_updates(policy.trainable_weights, policy.constraints, loss=loss_var)
            train_op = tf.group(*updates)
            return train_op, loss_var

        logger.log("Initializing optimization")
        train_op, loss_var = init_opt()
        logger.log("Initialized")

        with tf.Session() as sess:

            logger.log("Initializing variables")
            tensor_utils.initialize_new_variables(sess)
            logger.log("Initialized")

            for epoch_idx in itertools.count():

                logger.log("Start training")
                loss_vals = []

                for _ in pyprind.prog_bar(range(n_updates_per_epoch)):
                    demo_paths, batch_obs, batch_actions = train_dataset.sample_bc_batch(
                        demo_batch_size,
                        per_demo_batch_size,
                        per_demo_full_traj
                    )
                    demo_vals = policy.process_demo_data(demo_paths)
                    feed_dict = dict()
                    feed_dict[obs_var] = batch_obs
                    feed_dict[action_var] = batch_actions
                    feed_dict[K.learning_phase()] = 1
                    for k, v in demo_vars.items():
                        feed_dict[v] = demo_vals[k]
                    _, loss_val = sess.run([train_op, loss_var], feed_dict=feed_dict)
                    loss_vals.append(loss_val)

                logger.log("Finished training")

                logger.record_tabular('Epoch', epoch_idx)
                logger.record_tabular('Loss', np.mean(loss_vals))

                if evaluate_policy:
                    logger.log("Sampling on-policy trajectory")

                    # Evaluate under each task... This can be pretty time consuming
                    def evaluate(dataset, scope):
                        if dataset.n_tasks > 0:
                            n_paths = 0
                            n_success_paths = 0
                            for task_id, eval_env in dataset.envs.items():
                                logger.log("Sampling on-policy trajectory for task {}".format(task_id))
                                with logger.tabular_prefix("{}Task_{}|".format(scope, task_id)):
                                    policy.inform_task(
                                        task_id=task_id,
                                        env=dataset.envs[task_id],
                                        paths=dataset.paths[task_id],
                                        obs=dataset.obs[task_id],
                                    )
                                    eval_paths = fetch_utils.new_policy_paths(
                                        seeds=np.random.randint(low=0, high=np.iinfo(np.int32).max,
                                                                size=n_eval_paths_per_task),
                                        env=eval_env,
                                        policy=eval_policy,
                                        horizon=horizon,
                                        xinits=xinits,
                                    )
                                    n_paths += len(eval_paths)
                                    n_success_paths += len(list(filter(eval_env.wrapped_env._is_success, eval_paths)))
                                    eval_env.log_diagnostics(eval_paths)
                            logger.record_tabular('{}SuccessRate'.format(scope), n_success_paths / n_paths)

                    evaluate(train_dataset, 'Train')
                    evaluate(test_dataset, 'Test')

                logger.dump_tabular()
                logger.save_itr_params(
                    itr=epoch_idx,
                    params=dict(
                        env=template_env,
                        policy=policy,
                    )
                )

    return AttrDict(train=train)
