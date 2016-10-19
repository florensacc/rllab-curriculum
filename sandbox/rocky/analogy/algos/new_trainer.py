import hashlib
import os
import pickle
import tempfile
from collections import OrderedDict

from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc import console
import numpy as np
import tensorflow as tf
import pyprind
from rllab.sampler.utils import rollout
from sandbox.rocky.analogy.policies.apply_demo_policy import ApplyDemoPolicy
from sandbox.rocky.analogy.dataset import SupervisedDataset
from sandbox.rocky.analogy.policies.normalizing_policy import NormalizingPolicy
from sandbox.rocky.analogy.utils import unwrap
from sandbox.rocky.s3.resource_manager import resource_manager
from rllab.sampler.stateful_pool import singleton_pool
from rllab import config
import itertools
import random
import contextlib

from sandbox.rocky.tf.envs.vec_env_executor import VecEnvExecutor
from sandbox.rocky.tf.misc import tensor_utils


@contextlib.contextmanager
def set_seed_tmp(seed=None):
    if seed is None:
        yield
    else:
        state = random.getstate()
        np_state = np.random.get_state()
        random.seed(seed)
        np.random.seed(seed)
        yield
        np.random.set_state(np_state)
        random.setstate(state)


def collect_demo(G, demo_collector, demo_seed, analogy_seed, target_seed, env_cls, horizon):
    demo_env = env_cls(seed=demo_seed, target_seed=target_seed)
    analogy_env = env_cls(seed=analogy_seed, target_seed=target_seed)

    demo_path = demo_collector.collect_demo(env=demo_env, horizon=horizon)
    analogy_path = demo_collector.collect_demo(env=analogy_env, horizon=horizon)

    return demo_path, analogy_path, demo_seed, analogy_seed, target_seed


def vectorized_rollout_analogy(policy, demo_paths, analogy_envs, max_path_length):
    vec_env = VecEnvExecutor(envs=analogy_envs)
    obses = vec_env.reset()
    dones = np.asarray([True] * vec_env.num_envs)
    running_paths = [None] * vec_env.num_envs
    finished = np.asarray([False] * vec_env.num_envs)

    env_spec = analogy_envs[0].spec

    progbar = pyprind.ProgBar(vec_env.num_envs)

    paths = []

    policy.apply_demos(demo_paths)

    while not np.all(finished):
        policy.reset(dones)
        actions, agent_infos = policy.get_actions(obses)

        next_obses, rewards, dones, env_infos = vec_env.step(actions, max_path_length=max_path_length)

        agent_infos = tensor_utils.split_tensor_dict_list(agent_infos)
        env_infos = tensor_utils.split_tensor_dict_list(env_infos)

        if env_infos is None:
            env_infos = [dict() for _ in range(vec_env.num_envs)]
        if agent_infos is None:
            agent_infos = [dict() for _ in range(vec_env.num_envs)]

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
                    paths.append(dict(
                        observations=env_spec.observation_space.flatten_n(running_paths[idx]["observations"]),
                        actions=env_spec.action_space.flatten_n(running_paths[idx]["actions"]),
                        rewards=tensor_utils.stack_tensor_list(running_paths[idx]["rewards"]),
                        env_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
                        agent_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
                    ))
                running_paths[idx] = None

        obses = next_obses

    if progbar.active:
        progbar.stop()

    assert (len(paths) == len(analogy_envs))

    return paths


def rollout_analogy(policy, demo_paths, analogy_envs, max_path_length):
    paths = []
    progbar = pyprind.ProgBar(len(demo_paths))
    for path, env in zip(demo_paths, analogy_envs):
        paths.append(rollout(env=env, agent=ApplyDemoPolicy(policy, demo_path=path), max_path_length=max_path_length))
        progbar.update()
    if progbar.active:
        progbar.stop()
    return paths


class Trainer(Serializable):
    def __init__(
            self,
            policy,
            env_cls,
            demo_collector,
            shuffler=None,
            demo_cache_key=None,
            n_train_trajs=50,
            n_test_trajs=20,
            horizon=50,
            batch_size=10,
            n_epochs=100,
            n_passes_per_epoch=1,
            n_eval_trajs=10,
            learning_rate=1e-3,
            no_improvement_tolerance=5,
            skip_eval=False,
            plot=False,
            intertwined=False,
    ):
        Serializable.quick_init(self, locals())
        self.env_cls = env_cls
        self.demo_collector = demo_collector
        self.demo_cache_key = demo_cache_key
        self.shuffler = shuffler
        self.n_train_trajs = n_train_trajs
        self.n_test_trajs = n_test_trajs
        self.horizon = horizon
        self.policy = policy
        self.plot = plot
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_passes_per_epoch = n_passes_per_epoch
        self.n_eval_trajs = n_eval_trajs
        self.skip_eval = skip_eval
        self.learning_rate = learning_rate
        self.no_improvement_tolerance = no_improvement_tolerance
        self.intertwined = intertwined

    def eval_and_log(self, policy, data_dict):
        eval_paths = vectorized_rollout_analogy(
            policy, data_dict["demo_paths"], data_dict["analogy_envs"], max_path_length=self.horizon
        )

        returns = [np.sum(p["rewards"]) for p in eval_paths]
        logger.record_tabular('AverageReturn', np.mean(returns))
        logger.record_tabular('MaxReturn', np.max(returns))
        logger.record_tabular('MinReturn', np.min(returns))

        log_envs = list(map(unwrap, data_dict["analogy_envs"]))
        log_envs[0].log_analogy_diagnostics(eval_paths, log_envs)

    def collect_trajs(self, demo_seeds, analogy_seeds, target_seeds):
        progbar = pyprind.ProgBar(len(demo_seeds))
        data_list = []

        for data in singleton_pool.run_imap_unordered(
                collect_demo,
                [(self.demo_collector,) + tuple(seeds) + (self.env_cls, self.horizon)
                 for seeds in zip(demo_seeds, analogy_seeds, target_seeds)]
        ):
            progbar.update(force_flush=True)
            data_list.append(data)
        if progbar.active:
            progbar.stop()

        return zip(*data_list)

    def init_opt(self, env, policy, horizon, batch_size=None):

        obs_dim = env.observation_space.flat_dim
        action_dim = env.action_space.flat_dim

        # initialize list of variables
        demo_obs_vars = [
            tf.placeholder(dtype=tf.float32, shape=(batch_size, obs_dim), name="demo_obs_{0}".format(t))
            for t in range(horizon)
        ]
        demo_action_vars = [
            tf.placeholder(dtype=tf.float32, shape=(batch_size, action_dim), name="demo_action_{0}".format(t))
            for t in range(horizon)
        ]
        valid_vars = [
            tf.placeholder(dtype=tf.int32, shape=(batch_size,), name="valid_{0}".format(t))
            for t in range(horizon)
        ]
        analogy_obs_vars = [
            tf.placeholder(dtype=tf.float32, shape=(batch_size, obs_dim), name="analogy_obs_{0}".format(t))
            for t in range(horizon)
        ]
        analogy_action_vars = [
            tf.placeholder(dtype=tf.float32, shape=(batch_size, action_dim), name="analogy_action_{0}".format(t))
            for t in range(horizon)
        ]

        # demo_obs_var = env.observation_space.new_tensor_variable(name="demo_obs", extra_dims=2)
        # demo_action_var = env.action_space.new_tensor_variable(name="demo_actions", extra_dims=2)
        #
        # analogy_obs_var = env.observation_space.new_tensor_variable(name="analogy_obs", extra_dims=2)
        # analogy_action_var = env.action_space.new_tensor_variable(name="analogy_actions", extra_dims=2)
        # prev_action_var = tf.concat(
        #     1,
        #     [
        #         tf.zeros(tf.pack([tf.shape(analogy_action_var)[0], 1, env.action_space.flat_dim])),
        #         analogy_action_var[:, :-1, :],
        #     ]
        # )

        lr_var = tf.placeholder(dtype=tf.float32, shape=(), name="lr")

        policy_action_vars = policy.action_sym(
            demo_obs_vars=demo_obs_vars,
            demo_action_vars=demo_action_vars,
            analogy_obs_vars=analogy_obs_vars,
            valid_vars=valid_vars
        )


        # policy_action_var = policy.action_sym(
        #     analogy_obs_var,
        #     state_info_vars=dict(
        #         demo_obs=demo_obs_var,
        #         demo_action=demo_action_var,
        #         # prev_action=prev_action_var,
        #     ),
        #     fixed_horizon=horizon
        #     # phase='train'
        # )
        # test_policy_action_var = policy.action_sym(
        #     analogy_obs_var,
        #     state_info_vars=dict(
        #         demo_obs=demo_obs_var,
        #         demo_action=demo_action_var,
        #         # prev_action=prev_action_var,
        #     ),
        #     phase='test'
        # )
        loss_var = tf.reduce_mean(tf.square(analogy_action_var - policy_action_var))
        # test_loss_var = tf.reduce_mean(tf.square(analogy_action_var - test_policy_action_var))

        optimizer = tf.train.AdamOptimizer(learning_rate=lr_var)

        params = policy.get_params(trainable=True)

        grads_and_vars = optimizer.compute_gradients(loss_var, var_list=params)
        train_op = optimizer.apply_gradients(grads_and_vars)

        def to_feed(batch_dict):
            demo_obs = np.asarray([p["observations"] for p in batch_dict["demo_paths"]])
            demo_actions = np.asarray([p["actions"] for p in batch_dict["demo_paths"]])
            analogy_obs = np.asarray([p["observations"] for p in batch_dict["analogy_paths"]])
            analogy_actions = np.asarray([p["actions"] for p in batch_dict["analogy_paths"]])
            return {
                demo_obs_var: demo_obs,
                demo_action_var: demo_actions,
                analogy_obs_var: analogy_obs,
                analogy_action_var: analogy_actions,
            }

        def f_train(batch_dict, learning_rate):
            feed = to_feed(batch_dict)
            feed[lr_var] = learning_rate
            _, loss = tf.get_default_session().run(
                [train_op, loss_var],
                feed_dict=feed,
            )
            return loss

        def f_test_loss(batch_dict):
            return tf.get_default_session().run(
                loss_var,
                feed_dict=to_feed(batch_dict),
            )

        return dict(f_train=f_train, f_test_loss=f_test_loss)

    def collect_demos(self):

        n_trajs = self.n_train_trajs + self.n_test_trajs

        data_dict = None

        def mkdict():
            demo_seeds, analogy_seeds, target_seeds = np.random.randint(
                low=0, high=np.iinfo(np.int32).max, size=(3, n_trajs)
            )
            demo_paths, analogy_paths, demo_seeds, analogy_seeds, target_seeds = \
                self.collect_trajs(demo_seeds, analogy_seeds, target_seeds)
            return OrderedDict([
                ("demo_paths", np.asarray(demo_paths)),
                ("analogy_paths", np.asarray(analogy_paths)),
                ("demo_seeds", np.asarray(demo_seeds)),
                ("analogy_seeds", np.asarray(analogy_seeds)),
                ("target_seeds", np.asarray(target_seeds)),
            ])

        if self.demo_cache_key is None:
            return mkdict()

        def mkfile():
            nonlocal data_dict
            data_dict = mkdict()

            f = tempfile.NamedTemporaryFile()
            np.savez_compressed(f, **data_dict)
            resource_manager.register_file(self.demo_cache_key, f.name)
            f.close()

        file_name = resource_manager.get_file(resource_name=self.demo_cache_key, mkfile=mkfile)

        if data_dict is not None:
            return data_dict

        data = np.load(file_name)

        demo_paths = data["demo_paths"][:n_trajs]
        analogy_paths = data["analogy_paths"][:n_trajs]
        demo_seeds = data["demo_seeds"][:n_trajs]
        analogy_seeds = data["analogy_seeds"][:n_trajs]
        target_seeds = data["target_seeds"][:n_trajs]

        return OrderedDict([
            ("demo_paths", np.asarray(demo_paths)),
            ("analogy_paths", np.asarray(analogy_paths)),
            ("demo_seeds", np.asarray(demo_seeds)),
            ("analogy_seeds", np.asarray(analogy_seeds)),
            ("target_seeds", np.asarray(target_seeds)),
        ])

    def train(self):

        data_dict = self.collect_demos()

        logger.log("Processing data")

        # dataset = BucketDataset(data_dict)

        dataset = SupervisedDataset(
            inputs=list(data_dict.values()),
            input_keys=list(data_dict.keys()),
            train_batch_size=self.batch_size,
            train_ratio=self.n_train_trajs * 1.0 / (self.n_train_trajs + self.n_test_trajs),
            shuffler=self.shuffler,
        )

        horizon = np.max([len(p["rewards"]) for p in np.concatenate([
            data_dict["demo_paths"],
            data_dict[ "analogy_paths"]
        ])])
        # import ipdb; ipdb.set_trace()

        # processing of dataset: need to decide on bucket sizes, and group the trajectories according to these bucket
        #  sizes

        env = self.env_cls(seed=data_dict["demo_seeds"][0], target_seed=data_dict["target_seeds"][0])

        # let's check consistency

        logger.log("Constructing optimization problem")

        # policy = self.policy
        train_dict = dataset.train.input_dict
        test_dict = dataset.test.input_dict
        n_test = len(dataset.test.inputs[0])
        subsampled_train_dict = {k: v[:n_test] for k, v in train_dict.items()}

        logger.log("Generating envs for evaluation")
        subsampled_train_dict["analogy_envs"] = [
            self.env_cls(seed=analogy_seed, target_seed=target_seed)
            for analogy_seed, target_seed in zip(
                subsampled_train_dict["analogy_seeds"],
                subsampled_train_dict["target_seeds"],
            )
            ]
        test_dict["analogy_envs"] = [
            self.env_cls(seed=analogy_seed, target_seed=target_seed)
            for analogy_seed, target_seed in zip(
                test_dict["analogy_seeds"],
                test_dict["target_seeds"],
            )
            ]

        policy = self.policy

        # policy = NormalizingPolicy(
        #     self.policy,
        #     demo_paths=train_dict["demo_paths"],
        #     analogy_paths=train_dict["analogy_paths"],
        #     # normalize_obs=True,
        # )

        opt_info = self.init_opt(env, policy, horizon=horizon, batch_size=None)#self.batch_size)

        # Best average return achieved by the NN policy
        best_loss = np.inf
        # Best parameter for the NN policy
        best_params = None
        # Number of epochs without improvement compared to the best policy so far
        n_no_improvement = 0

        # Current learning rate
        learning_rate = self.learning_rate

        logger.log("Launching TF session")

        with tf.Session() as sess:
            logger.log("Initializing TF variables")
            sess.run(tf.initialize_all_variables())
            logger.log("Initialized")

            for epoch_idx in range(self.n_epochs):
                losses = []
                logger.log("Start epoch %d" % epoch_idx)

                # Skip training for the first epoch
                if epoch_idx > 0:
                    logger.log("Start training...")
                    progbar = pyprind.ProgBar(dataset.train.number_batches * self.n_passes_per_epoch)
                    for _ in range(self.n_passes_per_epoch):
                        for batch in dataset.train.iterate(return_dict=True):
                            loss = opt_info["f_train"](
                                batch_dict=batch,
                                learning_rate=learning_rate
                            )
                            losses.append(loss)
                            progbar.update()
                    if progbar.active:
                        progbar.stop()
                    logger.log("Finished")
                else:
                    logger.log("Skipped training for the 0th epoch, to collect initial test statistics")

                logger.log("Computing loss on test set")
                test_loss = opt_info["f_test_loss"](batch_dict=test_dict)
                logger.log("Computed")

                avg_loss = np.mean(losses)

                if avg_loss > best_loss:
                    n_no_improvement += 1
                else:
                    n_no_improvement = 0
                    best_loss = avg_loss
                    # collect best params
                    best_params = policy.get_param_values(trainable=True)

                logger.record_tabular('Epoch', epoch_idx)
                logger.record_tabular("LearningRate", learning_rate)
                logger.record_tabular("NoImprovementEpochs", n_no_improvement)
                logger.record_tabular('AverageTrainLoss', avg_loss)
                logger.record_tabular('AverageTestLoss', test_loss)
                logger.record_tabular('OracleAverageReturn', np.mean(
                    [np.sum(p["rewards"]) for p in test_dict["analogy_paths"]]
                ))

                if not self.skip_eval:
                    logger.log("Evaluating on subsampled training set...")
                    with logger.tabular_prefix('Train'):
                        self.eval_and_log(policy=policy, data_dict=subsampled_train_dict)
                    logger.log("Evaluating on test set...")
                    with logger.tabular_prefix('Test'):
                        self.eval_and_log(policy=policy, data_dict=test_dict)

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
                    # save a version of the environment
                    env=test_dict["analogy_envs"][-1],
                    trainer=self,
                )
                logger.save_itr_params(epoch_idx, save_params)
                logger.log("Saved")
