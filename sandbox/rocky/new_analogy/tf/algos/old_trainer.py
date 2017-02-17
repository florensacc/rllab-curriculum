from rllab.core.serializable import Serializable
from rllab.misc import logger
import numpy as np
import tensorflow as tf
import pyprind
from rllab.sampler.utils import rollout
from sandbox.rocky.analogy.policies.apply_demo_policy import ApplyDemoPolicy
from sandbox.rocky.analogy.utils import unwrap
from rllab.sampler.stateful_pool import singleton_pool
import itertools
import pickle
import random
import contextlib

from sandbox.rocky.tf.distributions.recurrent_categorical import RecurrentCategorical
from sandbox.rocky.tf.envs.vec_env_executor import VecEnvExecutor
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.spaces import Box, Discrete


class AnalogyDatasetGroup(object):
    def __init__(self, paths):
        self.paths = np.asarray(paths)
        path_task_ids = np.asarray([p["env_infos"]["task_id"][0] for p in paths])
        unique_task_ids = np.asarray(sorted(set(path_task_ids)))

        path_ids_by_task = dict()
        for task_id in unique_task_ids:
            path_ids_by_task[task_id] = np.where(path_task_ids == task_id)[0]

        self.path_task_ids = path_task_ids
        self.unique_task_ids = unique_task_ids
        self.path_ids_by_task = path_ids_by_task

    def sample_batch(self, batch_size):
        batch_path_ids = np.random.choice(np.arange(len(self.paths)), size=batch_size)
        batch_path_task_ids = self.path_task_ids[batch_path_ids]
        batch_paths = self.paths[batch_path_ids]

        batch_analogy_paths = np.asarray([None] * len(batch_paths))

        for task_id in self.unique_task_ids:
            matched_ids = np.where(batch_path_task_ids == task_id)[0]
            if len(matched_ids) > 0:
                analogy_path_ids = np.random.choice(self.path_ids_by_task[task_id], size=len(matched_ids))
                batch_analogy_paths[matched_ids] = self.paths[analogy_path_ids]

        return dict(
            demo_paths=batch_paths,
            analogy_paths=batch_analogy_paths,
        )


class AnalogyDataset(object):
    def __init__(self, paths, train_ratio):
        paths = np.asarray(paths)

        path_task_ids = np.asarray([p["env_infos"]["task_id"][0] for p in paths])
        unique_task_ids = np.asarray(sorted(set(path_task_ids)))

        train_paths = []
        test_paths = []

        for task_id in unique_task_ids:
            selected_path_ids = np.where(path_task_ids == task_id)[0]
            np.random.shuffle(selected_path_ids)
            n_train = int(train_ratio * len(selected_path_ids))
            train_paths.extend(
                paths[selected_path_ids[:n_train]]
            )
            test_paths.extend(
                paths[selected_path_ids[n_train:]]
            )

        self.train = AnalogyDatasetGroup(train_paths)
        self.test = AnalogyDatasetGroup(test_paths)

        # Split paths according to train_ratio within each task_id


# @contextlib.contextmanager
# def set_seed_tmp(seed=None):
#     if seed is None:
#         yield
#     else:
#         state = random.getstate()
#         np_state = np.random.get_state()
#         random.seed(seed)
#         np.random.seed(seed)
#         yield
#         np.random.set_state(np_state)
#         random.setstate(state)
#
#
# def collect_demo(G, demo_collector, demo_seed, analogy_seed, target_seed, env_cls, horizon):
#     demo_env = env_cls(seed=demo_seed, target_seed=target_seed)
#     analogy_env = env_cls(seed=analogy_seed, target_seed=target_seed)
#
#     demo_path = demo_collector.collect_demo(env=demo_env, horizon=horizon)
#     analogy_path = demo_collector.collect_demo(env=analogy_env, horizon=horizon)
#
#     return demo_path, analogy_path, demo_seed, analogy_seed, target_seed
#
#
def vectorized_rollout_analogy(envs, policy, paths, max_path_length):
    vec_env = VecEnvExecutor(envs)
    dones = np.asarray([True] * vec_env.n_envs)
    obses = vec_env.reset(dones)
    running_paths = [None] * vec_env.n_envs
    finished = np.asarray([False] * vec_env.n_envs)
    paths = np.asarray(paths)

    env_spec = envs[0].spec

    progbar = pyprind.ProgBar(vec_env.n_envs)

    sampled_paths = []

    env_task_ids = [env.wrapped_env.conopt_env.task_id for env in envs]
    path_task_ids = [p["env_infos"]["task_id"][0] for p in paths]
    demo_paths = np.asarray([None] * len(env_task_ids))
    for task_id in set(path_task_ids):
        to_set = np.where(env_task_ids == task_id)[0]
        demo_paths[to_set] = paths[np.random.choice(
            np.where(path_task_ids == task_id)[0],
            size=len(to_set)
        )]

    policy.apply_demos(demo_paths)

    while not np.all(finished):
        policy.reset(dones)
        actions, agent_infos = policy.get_actions(obses)

        next_obses, rewards, dones, env_infos = vec_env.step(actions, max_path_length=max_path_length)

        agent_infos = tensor_utils.split_tensor_dict_list(agent_infos)
        env_infos = tensor_utils.split_tensor_dict_list(env_infos)

        if env_infos is None:
            env_infos = [dict() for _ in range(vec_env.n_envs)]
        if agent_infos is None:
            agent_infos = [dict() for _ in range(vec_env.n_envs)]

        for idx, observation, action, reward, env_info, agent_info, done in zip(itertools.count(), obses, actions,
                                                                                rewards, env_infos, agent_infos,
                                                                                dones):
            if running_paths[idx] is None:
                running_paths[idx] = dict(
                    observations=[],
                    actions=[],
                    rewards=[],
                    env_infos=[],
                    agent_infos=[],
                )
            running_paths[idx]["observations"].append(observation)
            running_paths[idx]["actions"].append(action)
            running_paths[idx]["rewards"].append(reward)
            running_paths[idx]["env_infos"].append(env_info)
            running_paths[idx]["agent_infos"].append(agent_info)
            if done:
                if not finished[idx]:
                    finished[idx] = True
                    progbar.update()
                    sampled_paths.append(dict(
                        observations=env_spec.observation_space.flatten_n(running_paths[idx]["observations"]),
                        actions=env_spec.action_space.flatten_n(running_paths[idx]["actions"]),
                        rewards=tensor_utils.stack_tensor_list(running_paths[idx]["rewards"]),
                        env_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
                        agent_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
                    ))
                    # n_samples += len(running_paths[idx]["rewards"])
                    # if n_samples >= batch_size:
                    #     break
                running_paths[idx] = None

        obses = next_obses

    if progbar.active:
        progbar.stop()

    # assert (len(paths) == len(analogy_envs))

    return sampled_paths


# A simple example hopefully able to train a feed-forward network

class Trainer(Serializable):
    def __init__(
            self,
            env,
            policy,
            demo_path=None,
            paths=None,
            normalize=True,
            demo_cache_key=None,
            horizon=50,
            batch_size=10,
            n_epochs=100,
            n_passes_per_epoch=1,
            learning_rate=1e-3,
            train_ratio=0.9,
            no_improvement_tolerance=5,
            eval_samples=10000,
            eval_horizon=100,
            gradient_clipping=40,
            task_id_filter=None,
            threshold=None,
    ):
        Serializable.quick_init(self, locals())
        self.env = env
        self.demo_path = demo_path
        self.paths = paths
        self.train_ratio = train_ratio
        self.normalize = normalize
        self.demo_cache_key = demo_cache_key
        self.horizon = horizon
        self.policy = policy
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_passes_per_epoch = n_passes_per_epoch
        self.learning_rate = learning_rate
        self.no_improvement_tolerance = no_improvement_tolerance
        self.eval_samples = eval_samples
        self.eval_horizon = eval_horizon
        self.gradient_clipping = gradient_clipping
        self.task_id_filter = task_id_filter
        self.threshold = threshold

    def init_opt(self, env, policy):
        demo_obs_var = env.observation_space.new_tensor_variable(name="demo_obs", extra_dims=2)
        demo_actions_var = env.action_space.new_tensor_variable(name="demo_actions", extra_dims=2)
        demo_valids_var = tf.placeholder(dtype=tf.float32, shape=(None, None), name="demo_valids")

        analogy_obs_var = env.observation_space.new_tensor_variable(name="analogy_obs", extra_dims=2)
        analogy_actions_var = env.action_space.new_tensor_variable(name="analogy_actions", extra_dims=2)
        analogy_valids_var = tf.placeholder(dtype=tf.float32, shape=(None, None), name="analogy_valids")

        lr_var = tf.placeholder(dtype=tf.float32, shape=(), name="lr")

        train_policy_action_var = policy.action_sym(
            analogy_obs_var,
            state_info_vars=dict(
                demo_obs=demo_obs_var,
                demo_actions=demo_actions_var,
                demo_valids=demo_valids_var,
            ),
            phase='train'
        )
        test_policy_action_var = policy.action_sym(
            analogy_obs_var,
            state_info_vars=dict(
                demo_obs=demo_obs_var,
                demo_actions=demo_actions_var,
                demo_valids=demo_valids_var,
            ),
            phase='test'
        )

        def _loss_sym(action_var):
            if isinstance(env.action_space, Box):
                return tf.reduce_sum(
                    tf.reduce_sum(tf.square(analogy_actions_var - action_var), -1) * analogy_valids_var
                ) / tf.reduce_sum(analogy_valids_var)
            elif isinstance(env.action_space, Discrete):
                kl = RecurrentCategorical(dim=env.action_space.n).kl_sym(dict(prob=analogy_actions_var),
                                                                         dict(prob=action_var))
                return tf.reduce_sum(kl * analogy_valids_var) / tf.reduce_sum(analogy_valids_var)
            else:
                raise NotImplementedError

        train_loss_var = _loss_sym(train_policy_action_var)
        test_loss_var = _loss_sym(test_policy_action_var)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr_var)

        params = policy.get_params(trainable=True)

        grads = tf.gradients(train_loss_var, xs=params)

        if self.gradient_clipping is not None:
            grads, _ = tf.clip_by_global_norm(grads, clip_norm=self.gradient_clipping)

        train_op = optimizer.apply_gradients(zip(grads, params))

        def to_feed(batch_dict):
            # we need to pad it...

            demo_paths = batch_dict["demo_paths"]
            analogy_paths = batch_dict["analogy_paths"]

            max_demo_len = np.max([len(p["observations"]) for p in demo_paths])
            demo_obs = tensor_utils.pad_tensor_n(
                [p["observations"] for p in demo_paths], max_demo_len)
            demo_actions = tensor_utils.pad_tensor_n(
                [p["actions"] for p in demo_paths], max_demo_len)
            demo_valids = tensor_utils.pad_tensor_n(
                [np.ones((len(p["observations"]),)) for p in demo_paths], max_demo_len)

            max_analogy_len = np.max([len(p["observations"]) for p in analogy_paths])
            analogy_obs = tensor_utils.pad_tensor_n(
                [p["observations"] for p in analogy_paths], max_analogy_len)
            analogy_actions = tensor_utils.pad_tensor_n(
                [p["actions"] for p in analogy_paths], max_analogy_len)
            analogy_valids = tensor_utils.pad_tensor_n(
                [np.ones((len(p["observations"]),)) for p in analogy_paths], max_analogy_len)

            return {
                demo_obs_var: demo_obs,
                demo_actions_var: demo_actions,
                demo_valids_var: demo_valids,
                analogy_obs_var: analogy_obs,
                analogy_actions_var: analogy_actions,
                analogy_valids_var: analogy_valids,
            }

        def f_train(batch_dict, learning_rate):
            feed = to_feed(batch_dict)
            feed[lr_var] = learning_rate
            _, loss = tf.get_default_session().run(
                [train_op, train_loss_var],
                feed_dict=feed,
            )
            return loss

        def f_test_loss(batch_dict):
            return tf.get_default_session().run(
                test_loss_var,
                feed_dict=to_feed(batch_dict),
            )

        return dict(f_train=f_train, f_test_loss=f_test_loss)

    def collect_demos(self):
        logger.log("Loading data...")
        data = np.load(self.demo_path)
        exp_x = data["exp_x"]
        exp_u = data["exp_u"]
        exp_rewards = data["exp_rewards"]
        exp_task_ids = data["exp_task_ids"]
        logger.log("Loaded")
        paths = []
        for xs, us, rewards, task_id in zip(exp_x, exp_u, exp_rewards[:, :, 0], exp_task_ids):
            # Filter by performance
            if self.task_id_filter is None or task_id in self.task_id_filter:
                paths.append(
                    dict(
                        observations=xs,
                        actions=us,
                        rewards=rewards,
                        env_infos=dict(
                            task_id=np.asarray([task_id] * len(xs))
                        )
                    )
                )
        return np.asarray(paths)

    def train(self, sess=None):

        if self.paths is None:
            paths = self.collect_demos()
        else:
            paths = self.paths
        if self.threshold is not None:
            paths = [p for p in self.paths if p["rewards"][-1] >= self.threshold]

        env = self.env
        policy = self.policy

        opt_info = self.init_opt(env, policy)

        # Best average return achieved by the NN policy
        best_loss = np.inf
        # Best parameter for the NN policy
        best_params = None
        # Number of epochs without improvement compared to the best policy so far
        n_no_improvement = 0

        # Current learning rate
        learning_rate = self.learning_rate

        logger.log("Launching TF session")

        if sess is None:
            sess = tf.Session()
            sess.__enter__()
            logger.log("Initializing TF variables")
            tensor_utils.initialize_new_variables(sess=sess)
            logger.log("Initialized")

        dataset = AnalogyDataset(paths, train_ratio=self.train_ratio)
        n_train = len(dataset.train.paths)
        n_test = len(dataset.test.paths)

        logger.log("Initialize environments for testing")
        n_eval_envs = int(np.ceil(self.eval_samples / self.eval_horizon))
        eval_envs = [pickle.loads(pickle.dumps(env)) for _ in range(n_eval_envs)]

        for epoch_idx in range(self.n_epochs):
            logger.log("Start epoch %d" % epoch_idx)

            losses = []

            progbar = pyprind.ProgBar(n_train * self.n_passes_per_epoch)
            for _ in range(n_train * self.n_passes_per_epoch // self.batch_size):
                batch_dict = dataset.train.sample_batch(batch_size=self.batch_size)
                loss = opt_info["f_train"](batch_dict, learning_rate=learning_rate)
                losses.append(loss)
                progbar.update(self.batch_size)

            if progbar.active:
                progbar.stop()
            logger.log("Finished")

            logger.log("Computing loss on test set")

            progbar = pyprind.ProgBar(n_test)
            test_losses = []
            for _ in range(int(np.ceil(n_test / self.batch_size))):
                batch_dict = dataset.test.sample_batch(batch_size=self.batch_size)
                test_losses.append(opt_info["f_test_loss"](batch_dict))
                progbar.update(self.batch_size)
            if progbar.active:
                progbar.stop()
            test_loss = np.mean(test_losses)
            logger.log("Computed")

            avg_loss = np.mean(losses)

            policy_params = policy.get_param_values(trainable=True)

            if avg_loss > best_loss:
                n_no_improvement += 1
            else:
                n_no_improvement = 0
                best_loss = avg_loss
                # collect best params
                best_params = policy_params

            logger.record_tabular('Epoch', epoch_idx)
            logger.record_tabular("LearningRate", learning_rate)
            logger.record_tabular("NoImprovementEpochs", n_no_improvement)
            logger.record_tabular('AverageTrainLoss', avg_loss)
            logger.record_tabular('AverageTestLoss', test_loss)
            logger.record_tabular('NPolicyParams', len(policy_params))
            logger.record_tabular('PolicyParamNorm', np.linalg.norm(policy_params))
            logger.record_tabular('OracleAverageReturn', np.mean(
                [np.sum(p["rewards"]) for p in paths]
            ))

            # evaluate policy performance
            logger.log("Evaluating policy performance")
            eval_paths = vectorized_rollout_analogy(
                envs=eval_envs,
                policy=policy,
                paths=paths,
                max_path_length=self.eval_horizon,
            )
            logger.log("Finished evaluation")#Evaluating policy performance")

            returns = [np.sum(p["rewards"]) for p in eval_paths]
            logger.record_tabular_misc_stat(key='Return', values=returns, placement='front')#np.mean(returns))
            env.log_diagnostics(eval_paths)
            # logger.record_tabular('MaxReturn', np.max(returns))
            # logger.record_tabular('MinReturn', np.min(returns))

            #
            #     if not self.skip_eval:
            #         logger.log("Evaluating on subsampled training set...")
            #         with logger.tabular_prefix('Train'):
            #             self.eval_and_log(policy=policy, data_dict=subsampled_train_dict)
            #         logger.log("Evaluating on test set...")
            #         with logger.tabular_prefix('Test'):
            #             eval_paths = self.eval_and_log(policy=policy, data_dict=test_dict)
            #             if self.use_curriculum:
            #                 test_analogy_envs = list(map(unwrap, test_dict["analogy_envs"]))
            #                 if self.curriculum_criterion(eval_paths, test_analogy_envs):
            #                     # jump to the next curriculum level
            #                     curriculum_finished = True
            #
            logger.dump_tabular()

            if n_no_improvement >= self.no_improvement_tolerance:
                learning_rate *= 0.5
                logger.log("No improvement for %d epochs. Reducing learning rate to %f" % (n_no_improvement,
                                                                                           learning_rate))
                n_no_improvement = 0
                # restore to best params
                policy.set_param_values(best_params, trainable=True)

            logger.log("Saving itr params..")

            save_params = dict(
                policy=policy,
                env=env,
                trainer=self,
            )
            logger.save_itr_params(epoch_idx, save_params, use_cloudpickle=True)
            logger.log("Saved")
