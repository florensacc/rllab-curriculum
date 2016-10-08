from rllab.core.serializable import Serializable
from rllab.misc import logger
import numpy as np
import tensorflow as tf
import pyprind
from rllab.sampler.utils import rollout
from sandbox.rocky.analogy.policies.apply_demo_policy import ApplyDemoPolicy
from sandbox.rocky.analogy.dataset import SupervisedDataset
from sandbox.rocky.analogy.policies.normalizing_policy import NormalizingPolicy
from rllab.sampler.stateful_pool import singleton_pool
import itertools
import random
import contextlib

from sandbox.rocky.analogy.utils import unwrap


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


def collect_demo(G, demo_seed, analogy_seed, target_seed, env_cls, demo_policy_cls, horizon):
    demo_env = env_cls(seed=demo_seed, target_seed=target_seed)
    analogy_env = env_cls(seed=analogy_seed, target_seed=target_seed)
    demo_path = rollout(demo_env, demo_policy_cls(demo_env), max_path_length=horizon)
    analogy_path = rollout(analogy_env, demo_policy_cls(analogy_env), max_path_length=horizon)
    return demo_path, analogy_path, demo_env, analogy_env


# A simple example hopefully able to train a feed-forward network

class Trainer(Serializable):
    def __init__(
            self,
            policy,
            env_cls,
            demo_policy_cls,
            shuffler=None,
            n_train_trajs=50,
            n_test_trajs=20,
            horizon=50,
            batch_size=10,
            n_epochs=100,
            n_passes_per_epoch=1,
            n_eval_trajs=10,
            learning_rate=1e-3,
            no_improvement_tolerance=5,
            plot=False,
    ):
        Serializable.quick_init(self, locals())
        self.env_cls = env_cls
        self.demo_policy_cls = demo_policy_cls
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
        self.learning_rate = learning_rate
        self.no_improvement_tolerance = no_improvement_tolerance

    def train(self):

        demo_seeds, analogy_seeds, target_seeds = np.random.randint(
            low=0, high=np.iinfo(np.int32).max, size=(3, self.n_train_trajs + self.n_test_trajs)
        )

        logger.log("Collecting trajectories")
        progbar = pyprind.ProgBar(len(demo_seeds))

        data_list = []

        for data in singleton_pool.run_imap_unordered(
                collect_demo,
                [tuple(seeds) + (self.env_cls, self.demo_policy_cls, self.horizon)
                 for seeds in zip(demo_seeds, analogy_seeds, target_seeds)]
        ):
            progbar.update()
            data_list.append(data)

        demo_paths, analogy_paths, demo_envs, analogy_envs = zip(*data_list)

        if progbar.active:
            progbar.stop()

        logger.log("Processing data")

        all_data_pairs = [
            ("demo_paths", np.asarray(demo_paths)),
            ("analogy_paths", np.asarray(analogy_paths)),
            # These will be ignored during training since they appear last
            ("demo_envs", np.array(demo_envs)),
            ("analogy_envs", np.array(analogy_envs)),
        ]
        all_data_keys = [x[0] for x in all_data_pairs]
        all_data_vals = [x[1] for x in all_data_pairs]

        dataset = SupervisedDataset(
            inputs=all_data_vals,
            train_batch_size=self.batch_size,
            train_ratio=self.n_train_trajs * 1.0 / (self.n_train_trajs + self.n_test_trajs),
            shuffler=self.shuffler,
        )

        env = demo_envs[0]

        logger.log("Constructing optimization problem")
        policy = self.policy
        policy = NormalizingPolicy(
            self.policy,
            *dataset.train.inputs[:2]
        )

        demo_obs_var = env.observation_space.new_tensor_variable(name="demo_obs", extra_dims=2)
        demo_action_var = env.action_space.new_tensor_variable(name="demo_actions", extra_dims=2)

        analogy_obs_var = env.observation_space.new_tensor_variable(name="analogy_obs", extra_dims=2)
        analogy_action_var = env.action_space.new_tensor_variable(name="analogy_actions", extra_dims=2)

        lr_var = tf.placeholder(dtype=tf.float32, shape=(), name="lr")

        train_policy_action_var = policy.action_sym(
            analogy_obs_var,
            state_info_vars=dict(
                demo_obs=demo_obs_var,
                demo_action=demo_action_var
            ),
            phase='train'
        )
        test_policy_action_var = policy.action_sym(
            analogy_obs_var,
            state_info_vars=dict(
                demo_obs=demo_obs_var,
                demo_action=demo_action_var
            ),
            phase='test'
        )
        train_loss_var = tf.reduce_mean(tf.square(analogy_action_var - train_policy_action_var))
        test_loss_var = tf.reduce_mean(tf.square(analogy_action_var - test_policy_action_var))

        optimizer = tf.train.AdamOptimizer(learning_rate=lr_var)

        params = policy.get_params(trainable=True)

        grads_and_vars = optimizer.compute_gradients(train_loss_var, var_list=params)
        train_op = optimizer.apply_gradients(grads_and_vars)

        # Best average return achieved by the NN policy
        best_loss = np.inf
        # Best parameter for the NN policy
        best_params = None
        # Number of epochs without improvement compared to the best policy so far
        n_no_improvement = 0

        # Current learning rate
        learning_rate = self.learning_rate

        def to_feed(batch):
            batch_dict = dict(zip(all_data_keys, batch))
            demo_obs = np.asarray([p["observations"] for p in batch_dict["demo_paths"]])
            demo_actions = np.asarray([p["actions"] for p in batch_dict["demo_paths"]])
            analogy_obs = np.asarray([p["observations"] for p in batch_dict["analogy_paths"]])
            analogy_actions = np.asarray([p["actions"] for p in batch_dict["analogy_paths"]])
            return {
                demo_obs_var: demo_obs,
                demo_action_var: demo_actions,
                analogy_obs_var: analogy_obs,
                analogy_action_var: analogy_actions,
                lr_var: learning_rate,
            }

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
                        for batch in dataset.train.iterate():
                            _, loss = sess.run(
                                [train_op, train_loss_var],
                                feed_dict=to_feed(batch),
                            )
                            losses.append(loss)
                            progbar.update()
                    if progbar.active:
                        progbar.stop()
                    logger.log("Finished")
                else:
                    logger.log("Skipped training for the 0th epoch, to collect initial test statistics")

                test_loss = sess.run(
                    test_loss_var,
                    feed_dict=to_feed(dataset.test.inputs),
                )

                test_dict = dict(zip(all_data_keys, dataset.test.inputs))

                # Evaluate performance

                eval_paths = []

                for idx, demo_path, analogy_env in zip(
                        itertools.count(),
                        test_dict["demo_paths"],
                        test_dict["analogy_envs"],
                ):
                    eval_paths.append(rollout(
                        analogy_env, ApplyDemoPolicy(policy, demo_path), max_path_length=self.horizon,
                        animated=self.plot and idx == 0,
                    ))

                # import ipdb; ipdb.set_trace()

                if self.plot:
                    rollout(
                        analogy_env, ApplyDemoPolicy(policy, demo_path), max_path_length=self.horizon,
                        animated=self.plot and idx == 0,
                    )

                returns = [np.sum(p["rewards"]) for p in eval_paths]

                avg_loss = np.mean(losses)

                # avg_train_loss = np.mean(train_losses)
                if avg_loss > best_loss:
                    n_no_improvement += 1
                else:
                    n_no_improvement = 0
                    best_loss = avg_loss
                    # collect best params
                    best_params = sess.run(params)

                logger.record_tabular('Epoch', epoch_idx)
                logger.record_tabular("LearningRate", learning_rate)
                logger.record_tabular("NoImprovementEpochs", n_no_improvement)
                logger.record_tabular('AverageTrainLoss', avg_loss)
                logger.record_tabular('AverageTestLoss', test_loss)
                logger.record_tabular('AverageReturn', np.mean(returns))
                logger.record_tabular('MaxReturn', np.max(returns))
                logger.record_tabular('MinReturn', np.min(returns))
                logger.record_tabular('OracleAverageReturn', np.mean(
                    [np.sum(p["rewards"]) for p in test_dict["analogy_paths"]]
                ))
                log_env = unwrap(analogy_envs[-1])
                log_envs = map(unwrap, test_dict["analogy_envs"])
                log_env.log_analogy_diagnostics(eval_paths, log_envs)

                logger.dump_tabular()

                if n_no_improvement >= self.no_improvement_tolerance:
                    learning_rate *= 0.5
                    logger.log("No improvement for %d epochs. Reducing learning rate to %f" % (n_no_improvement,
                                                                                               learning_rate))
                    n_no_improvement = 0
                    # restore to best params
                    sess.run([tf.assign(p, pv) for p, pv in zip(params, best_params)])

                logger.log("Saving itr params..")

                save_params = dict(
                    policy=policy,
                    # save a version of the environment
                    env=analogy_envs[-1],
                    trainer=self,
                )
                logger.save_itr_params(epoch_idx, save_params)
                logger.log("Saved")
