import tempfile
from collections import OrderedDict

from rllab.core.serializable import Serializable
from rllab.misc import logger
import numpy as np
import tensorflow as tf
import pyprind
from rllab.sampler.utils import rollout
from sandbox.rocky.analogy.policies.apply_demo_policy import ApplyDemoPolicy
from sandbox.rocky.analogy.dataset import SupervisedDataset
from sandbox.rocky.analogy.policies.normalizing_policy import NormalizingPolicy
from sandbox.rocky.analogy.utils import unwrap
from rllab.sampler.stateful_pool import singleton_pool
import itertools
import random
import contextlib

from sandbox.rocky.s3.resource_manager import resource_manager
from sandbox.rocky.tf.distributions.recurrent_categorical import RecurrentCategorical
from sandbox.rocky.tf.envs.vec_env_executor import VecEnvExecutor
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.spaces import Box, Discrete


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


# A simple example hopefully able to train a feed-forward network

class Trainer(Serializable):
    def __init__(
            self,
            policy,
            env_cls,
            demo_collector,
            generalization_env_cls_list=None,
            normalize=True,
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
            use_curriculum=False,
            curriculum_env_cls_list=None,
            curriculum_criterion=None,
            gradient_clipping=40,
    ):
        Serializable.quick_init(self, locals())
        self.env_cls = env_cls
        self.generalization_env_cls_list = generalization_env_cls_list
        self.demo_collector = demo_collector
        self.normalize = normalize
        self.demo_cache_key = demo_cache_key
        # self.demo_policy_cls = demo_policy_cls
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
        self.use_curriculum = use_curriculum
        self.curriculum_env_cls_list = curriculum_env_cls_list
        self.curriculum_criterion = curriculum_criterion
        self.gradient_clipping = gradient_clipping

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
        return eval_paths

    def collect_trajs(self, demo_seeds, analogy_seeds, target_seeds, env_cls):
        progbar = pyprind.ProgBar(len(demo_seeds))
        data_list = []

        for data in singleton_pool.run_imap_unordered(
                collect_demo,
                [(self.demo_collector,) + tuple(seeds) + (env_cls, self.horizon)
                 for seeds in zip(demo_seeds, analogy_seeds, target_seeds)]
        ):
            progbar.update(force_flush=True)
            data_list.append(data)
        if progbar.active:
            progbar.stop()

        return zip(*data_list)

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

        gvs = optimizer.compute_gradients(train_loss_var, var_list=params)

        if self.gradient_clipping is not None:
            capped_gvs = [
                (tf.clip_by_value(grad, -self.gradient_clipping, self.gradient_clipping), var)
                if grad is not None else (grad, var)
                for grad, var in gvs
                ]
        else:
            capped_gvs = gvs

        train_op = optimizer.apply_gradients(capped_gvs)

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

    def collect_demos(self, env_cls):

        assert self.demo_cache_key is not None

        env = env_cls()
        if hasattr(self.demo_cache_key, "__call__"):
            demo_cache_key = self.demo_cache_key(self, unwrap(env), self.policy)
        else:
            demo_cache_key = self.demo_cache_key

        n_trajs = self.n_train_trajs + self.n_test_trajs

        data_dict = None

        def mkdict():
            demo_seeds, analogy_seeds, target_seeds = np.random.randint(
                low=0, high=np.iinfo(np.int32).max, size=(3, n_trajs)
            )
            demo_paths, analogy_paths, demo_seeds, analogy_seeds, target_seeds = \
                self.collect_trajs(demo_seeds, analogy_seeds, target_seeds, env_cls)
            return OrderedDict([
                ("demo_paths", np.asarray(demo_paths)),
                ("analogy_paths", np.asarray(analogy_paths)),
                ("demo_seeds", np.asarray(demo_seeds)),
                ("analogy_seeds", np.asarray(analogy_seeds)),
                ("target_seeds", np.asarray(target_seeds)),
            ])

        def mkfile():
            nonlocal data_dict
            data_dict = mkdict()

            f = tempfile.NamedTemporaryFile()
            np.savez_compressed(f, **data_dict)
            resource_manager.register_file(demo_cache_key, f.name)
            f.close()

        file_name = resource_manager.get_file(resource_name=demo_cache_key, mkfile=mkfile)

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

        if self.use_curriculum:
            curriculum_env_cls_list = self.curriculum_env_cls_list
        else:
            curriculum_env_cls_list = [self.env_cls]

        policy = None
        opt_info = None
        sess = None
        epoch_idx = 0

        for curriculum_idx, env_cls in enumerate(curriculum_env_cls_list):

            env = env_cls(seed=0, target_seed=0)

            data_dict = self.collect_demos(env_cls)

            logger.log("Processing data")

            dataset = SupervisedDataset(
                inputs=list(data_dict.values()),
                input_keys=list(data_dict.keys()),
                train_batch_size=self.batch_size,
                train_ratio=self.n_train_trajs * 1.0 / (self.n_train_trajs + self.n_test_trajs),
                shuffler=self.shuffler,
            )

            # let's check consistency

            logger.log("Constructing optimization problem")

            # policy = self.policy
            train_dict = dataset.train.input_dict
            test_dict = dataset.test.input_dict
            n_test = len(dataset.test.inputs[0])
            subsampled_train_dict = {k: v[:n_test] for k, v in train_dict.items()}

            logger.log("Generating envs for evaluation")
            subsampled_train_dict["analogy_envs"] = [
                env_cls(seed=analogy_seed, target_seed=target_seed)
                for analogy_seed, target_seed in zip(
                    subsampled_train_dict["analogy_seeds"],
                    subsampled_train_dict["target_seeds"],
                )
                ]
            test_dict["analogy_envs"] = [
                env_cls(seed=analogy_seed, target_seed=target_seed)
                for analogy_seed, target_seed in zip(
                    test_dict["analogy_seeds"],
                    test_dict["target_seeds"],
                )
                ]
            if self.generalization_env_cls_list is not None:
                logger.log("Generalizing envs for generalization evaluation")
                generalization_envs = []
                for tag, gen_env_cls in self.generalization_env_cls_list:
                    gen_dict = dict(
                        demo_paths=[],
                        analogy_paths=[],
                        demo_seeds=[],
                        analogy_seeds=[],
                        target_seeds=[],
                        analogy_envs=[],
                    )
                    for demo_seed, analogy_seed, target_seed in zip(test_dict["demo_seeds"], test_dict[
                        "analogy_seeds"], test_dict["target_seeds"]):
                        demo_env = gen_env_cls(seed=demo_seed, target_seed=target_seed)
                        analogy_env = gen_env_cls(seed=analogy_seed, target_seed=target_seed)
                        demo_path = self.demo_collector.collect_demo(env=demo_env, horizon=self.horizon)
                        analogy_path = self.demo_collector.collect_demo(env=analogy_env, horizon=self.horizon)
                        gen_dict["demo_paths"].append(demo_path)
                        gen_dict["analogy_paths"].append(analogy_path)
                        gen_dict["demo_seeds"].append(demo_seed)
                        gen_dict["analogy_seeds"].append(analogy_seed)
                        gen_dict["target_seeds"].append(target_seed)
                        gen_dict["analogy_envs"].append(analogy_env)
                    generalization_envs.append((tag, gen_dict))
            else:
                generalization_envs = []

            if policy is None:
                if self.normalize:
                    policy = NormalizingPolicy(
                        self.policy,
                        demo_paths=train_dict["demo_paths"],
                        analogy_paths=train_dict["analogy_paths"],
                    )
                else:
                    policy = self.policy

            if opt_info is None:
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
                sess.run(tf.initialize_all_variables())
                logger.log("Initialized")

            curriculum_epoch_idx = 0
            curriculum_finished = False

            while epoch_idx < self.n_epochs:
                losses = []
                logger.log("Start epoch %d" % epoch_idx)

                # Skip training for the first epoch
                if curriculum_epoch_idx > 0:
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

                policy_params = policy.get_param_values(trainable=True)

                if avg_loss > best_loss:
                    n_no_improvement += 1
                else:
                    n_no_improvement = 0
                    best_loss = avg_loss
                    # collect best params
                    best_params = policy_params

                logger.record_tabular('Epoch', epoch_idx)
                logger.record_tabular('CurriculumIdx', curriculum_idx)
                logger.record_tabular("LearningRate", learning_rate)
                logger.record_tabular("NoImprovementEpochs", n_no_improvement)
                logger.record_tabular('AverageTrainLoss', avg_loss)
                logger.record_tabular('AverageTestLoss', test_loss)
                logger.record_tabular('NPolicyParams', len(policy_params))
                logger.record_tabular('AvgPolicyParamNorm', np.linalg.norm(policy_params) / len(policy_params))
                logger.record_tabular('OracleAverageReturn', np.mean(
                    [np.sum(p["rewards"]) for p in test_dict["analogy_paths"]]
                ))

                if not self.skip_eval:
                    logger.log("Evaluating on subsampled training set...")
                    with logger.tabular_prefix('Train'):
                        self.eval_and_log(policy=policy, data_dict=subsampled_train_dict)
                    logger.log("Evaluating on test set...")
                    with logger.tabular_prefix('Test'):
                        eval_paths = self.eval_and_log(policy=policy, data_dict=test_dict)
                        if self.use_curriculum:
                            test_analogy_envs = list(map(unwrap, test_dict["analogy_envs"]))
                            if self.curriculum_criterion(eval_paths, test_analogy_envs):
                                # jump to the next curriculum level
                                curriculum_finished = True
                    if len(generalization_envs) > 0:
                        logger.log("Evaluating on generalization envs...")
                        for tag, gen_dict in generalization_envs:
                            with logger.tabular_prefix('Test[%s]' % tag):
                                self.eval_and_log(policy=policy, data_dict=gen_dict)

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
                logger.save_itr_params(epoch_idx, save_params, use_cloudpickle=True)
                logger.log("Saved")

                epoch_idx += 1
                curriculum_epoch_idx += 1
                if curriculum_finished:
                    break
