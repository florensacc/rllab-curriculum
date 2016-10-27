import time
from rllab.algos.base import RLAlgorithm
import rllab.misc.logger as logger
import rllab.plotter as plotter
from sandbox.rocky.tf.policies.base import Policy
import tensorflow as tf
import numpy as np

from sandbox.rocky.neural_learner.sample_processors.default_sample_processor import DefaultSampleProcessor
from sandbox.rocky.neural_learner.samplers.vectorized_sampler import VectorizedSampler


class BatchPolopt(RLAlgorithm):
    """
    Base class for batch sampling-based policy optimization methods.
    This includes various policy gradient methods like vpg, npg, ppo, trpo, etc.
    """

    def __init__(
            self,
            env,
            policy,
            baseline,
            scope=None,
            n_itr=500,
            start_itr=0,
            batch_size=5000,
            batch_size_schedule=None,
            max_path_length=500,
            max_path_length_schedule=None,
            discount=0.99,
            gae_lambda=1,
            plot=False,
            pause_for_plot=False,
            center_adv=True,
            positive_adv=False,
            store_paths=False,
            whole_paths=True,
            sampler=None,
            sample_processor=None,
            post_evals=None,
            n_vectorized_envs=None,
            **kwargs
    ):
        """
        :param env: Environment
        :param policy: Policy
        :type policy: Policy
        :param baseline: Baseline
        :param scope: Scope for identifying the algorithm. Must be specified if running multiple algorithms
        simultaneously, each using different environments and policies
        :param n_itr: Number of iterations.
        :param start_itr: Starting iteration.
        :param batch_size: Number of samples per iteration.
        :param max_path_length: Maximum length of a single rollout.
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param plot: Plot evaluation run after each iteration.
        :param pause_for_plot: Whether to pause before contiuing when plotting.
        :param center_adv: Whether to rescale the advantages so that they have mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are always positive. When used in
        conjunction with center_adv the advantages will be standardized before shifting.
        :param store_paths: Whether to save all paths data to the snapshot.
        :return:
        """
        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.scope = scope
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.batch_size = batch_size
        self.batch_size_schedule = batch_size_schedule
        self.max_path_length = max_path_length
        self.max_path_length_schedule = max_path_length_schedule
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.whole_paths = whole_paths

        if sampler is None:
            assert self.policy.vectorized
            if n_vectorized_envs is None:
                n_envs = max(1, int(np.ceil(batch_size / max_path_length)))
            else:
                n_envs = n_vectorized_envs
            sampler = VectorizedSampler(env=env, policy=policy, n_envs=n_envs)

        self.sampler = sampler

        if sample_processor is None:
            sample_processor = DefaultSampleProcessor(self)
        self.sample_procesor = sample_processor

        if post_evals is None:
            post_evals = []

        for post_eval in post_evals:
            if "env" not in post_eval:
                post_eval["env"] = env
            if "policy" not in post_eval:
                post_eval["policy"] = policy
            if "sampler" not in post_eval:
                assert post_eval["policy"].vectorized
                if n_vectorized_envs is None:
                    n_envs = max(1, int(np.ceil(post_eval["batch_size"] / max_path_length)))
                else:
                    n_envs = n_vectorized_envs
                post_eval["sampler"] = VectorizedSampler(
                    env=post_eval["env"],
                    policy=post_eval["policy"],
                    n_envs=n_envs
                )

        self.post_evals = post_evals

        self.init_opt()

    def start_worker(self):
        self.sampler.start_worker()
        for post_eval in self.post_evals:
            post_eval["sampler"].start_worker()
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    def shutdown_worker(self):
        self.sampler.shutdown_worker()
        for post_eval in self.post_evals:
            post_eval["sampler"].shutdown_worker()

    def obtain_samples(self, itr):
        if self.max_path_length_schedule is not None:
            max_path_length = self.max_path_length_schedule[itr]
        else:
            max_path_length = self.max_path_length
        if self.batch_size_schedule is not None:
            batch_size = self.batch_size_schedule[itr]
        else:
            batch_size = self.batch_size
        return self.sampler.obtain_samples(
            itr,
            max_path_length=max_path_length,
            batch_size=batch_size
        )

    def post_eval_policy(self, itr):
        for post_eval in self.post_evals:
            label = post_eval["label"]
            sampler = post_eval["sampler"]
            batch_size = post_eval["batch_size"]
            with logger.prefix(label + " | "), logger.tabular_prefix(label + "|"):
                paths = sampler.obtain_samples(
                    itr,
                    max_path_length=self.max_path_length,
                    batch_size=batch_size
                )
                returns = [sum(p["rewards"]) for p in paths]
                logger.record_tabular("NumTrajs", len(paths))
                logger.record_tabular_misc_stat('Return', returns, placement='front')
                post_eval["env"].log_diagnostics(paths)
                # log statistics for these paths

    def process_samples(self, itr, paths):
        return self.sample_procesor.process_samples(itr, paths)

    def train(self):
        with tf.Session() as sess:
            # writer = tf.train.SummaryWriter(logger.get_tf_summary_dir(), sess.graph)
            # logger.set_tf_summary_writer(writer)
            # summary_op = tf.merge_all_summaries()

            sess.run(tf.initialize_all_variables())

            logger.log("Starting worker...")
            self.start_worker()
            logger.log("Worker started")
            start_time = time.time()
            for itr in range(self.start_itr, self.n_itr):
                itr_start_time = time.time()
                with logger.prefix('itr #%d | ' % itr):
                    logger.log("Obtaining samples...")
                    paths = self.obtain_samples(itr)
                    logger.log("Processing samples...")
                    samples_data = self.process_samples(itr, paths)
                    logger.log("Logging diagnostics...")
                    self.log_diagnostics(paths)
                    logger.log("Optimizing policy...")
                    self.optimize_policy(itr, samples_data)
                    logger.log("Saving snapshot...")
                    params = self.get_itr_snapshot(itr, samples_data)  # , **kwargs)
                    if self.store_paths:
                        params["paths"] = samples_data["paths"]
                    logger.save_itr_params(itr, params)
                    logger.log("Saved")
                    logger.log("Evaluating on additional environments")
                    self.post_eval_policy(itr)
                    logger.log("Evaluated")
                    logger.record_tabular('Time', time.time() - start_time)
                    logger.record_tabular('ItrTime', time.time() - itr_start_time)
                    logger.dump_tabular(with_prefix=False)
                    if self.plot:
                        self.update_plot()
                        if self.pause_for_plot:
                            input("Plotting evaluation run: Press Enter to "
                                  "continue...")
        self.shutdown_worker()

    def log_diagnostics(self, paths):
        self.env.log_diagnostics(paths)
        self.policy.log_diagnostics(paths)
        self.baseline.log_diagnostics(paths)

    def init_opt(self):
        """
        Initialize the optimization procedure. If using tensorflow, this may
        include declaring all the variables and compiling functions
        """
        raise NotImplementedError

    def get_itr_snapshot(self, itr, samples_data):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """
        raise NotImplementedError

    def optimize_policy(self, itr, samples_data):
        raise NotImplementedError

    def update_plot(self):
        if self.plot:
            plotter.update_plot(self.policy, self.max_path_length)
