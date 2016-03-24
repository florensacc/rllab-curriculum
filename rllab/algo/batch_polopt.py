import numpy as np
from rllab.algo.base import RLAlgorithm
from rllab.sampler import parallel_sampler
from rllab.misc import autoargs
from rllab.misc import special
from rllab.misc import tensor_utils
from rllab.algo import util
import rllab.misc.logger as logger
import rllab.plotter as plotter


class BatchPolopt(RLAlgorithm):
    """
    Base class for batch sampling-based policy optimization methods.
    This includes various policy gradient methods like vpg, npg, ppo, trpo, etc.
    """

    def __init__(
            self,
            n_itr=500,
            start_itr=0,
            batch_size=5000,
            max_path_length=500,
            discount=0.99,
            gae_lambda=1,
            plot=False,
            pause_for_plot=False,
            whole_paths=True,
            center_adv=True,
            positive_adv=False,
            store_paths=False,
            **kwargs
    ):
        """
        :param n_itr: Number of iterations.
        :param start_itr: Starting iteration.
        :param batch_size: Number of samples per iteration.
        :param max_path_length: Maximum length of a single rollout.
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param plot: Plot evaluation run after each iteration.
        :param pause_for_plot: Whether to pause before contiuing when plotting.
        :param whole_paths: Make sure that the samples contain whole trajectories, even if the actual batch size is
        slightly larger than the specified batch_size.
        :param center_adv: Whether to rescale the advantages so that they have mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are always positive. When used in
        conjunction with center_adv the advantages will be standardized before shifting.
        :param store_paths: Whether to save all paths data to the snapshot.
        :return:
        """
        self._n_itr = n_itr
        self._start_itr = start_itr
        self._batch_size = batch_size
        self._max_path_length = max_path_length
        self._discount = discount
        self._gae_lambda = gae_lambda
        self._plot = plot
        self._pause_for_plot = pause_for_plot
        self._whole_paths = whole_paths
        self._center_adv = center_adv
        self._positive_adv = positive_adv
        self._store_paths = store_paths

    def start_worker(self, env, policy, baseline):
        parallel_sampler.populate_task(env, policy)
        if self._plot:
            plotter.init_plot(env, policy)

    def shutdown_worker(self):
        pass

    def train(self, env, policy, baseline, **kwargs):
        self.start_worker(env, policy, baseline)
        opt_info = self.init_opt(env.spec, policy, baseline)
        for itr in xrange(self._start_itr, self._n_itr):
            logger.push_prefix('itr #%d | ' % itr)
            paths = self.obtain_samples(itr, env, policy, **kwargs)
            samples_data = self.process_samples(itr, paths, env.spec, policy, baseline, **kwargs)
            env.log_diagnostics(paths)
            policy.log_diagnostics(paths)
            baseline.log_diagnostics(paths)
            opt_info = self.optimize_policy(
                itr, policy, samples_data, opt_info, **kwargs)
            logger.log("saving snapshot...")
            params = self.get_itr_snapshot(
                itr, env, policy, baseline, samples_data, opt_info, **kwargs)
            if self._store_paths:
                params["paths"] = samples_data["paths"]
            logger.save_itr_params(itr, params)
            logger.log("saved")
            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()
            if self._plot:
                self.update_plot(policy)
                if self._pause_for_plot:
                    raw_input("Plotting evaluation run: Press Enter to "
                              "continue...")
        self.shutdown_worker()

    def init_opt(self, env_spec, policy, baseline):
        """
        Initialize the optimization procedure. If using theano / cgt, this may
        include declaring all the variables and compiling functions
        """
        raise NotImplementedError

    def get_itr_snapshot(self, itr, env, policy, baseline, samples_data,
                         opt_info, **kwargs):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """
        raise NotImplementedError

    def optimize_policy(self, itr, policy, samples_data, opt_info):
        raise NotImplementedError

    def update_plot(self, policy):
        if self._plot:
            plotter.update_plot(policy, self._max_path_length)

    def obtain_samples(self, itr, env, policy, **kwargs):
        cur_params = policy.get_param_values()

        parallel_sampler.request_samples(
            policy_params=cur_params,
            max_samples=self._batch_size,
            max_path_length=self._max_path_length,
            whole_paths=self._whole_paths,
        )

        return parallel_sampler.collect_paths()

    def process_samples(self, itr, paths, env_spec, policy, baseline, **kwargs):

        baselines = []
        returns = []
        for path in paths:
            path_baselines = np.append(baseline.predict(path), 0)
            deltas = path["rewards"] + \
                     self._discount * path_baselines[1:] - \
                     path_baselines[:-1]
            path["advantages"] = special.discount_cumsum(
                deltas, self._discount * self._gae_lambda)
            path["returns"] = special.discount_cumsum(path["rewards"], self._discount)
            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])

        if not policy.recurrent:
            observations = tensor_utils.concat_tensor_list([path["observations"] for path in paths])
            actions = tensor_utils.concat_tensor_list([path["actions"] for path in paths])
            advantages = tensor_utils.concat_tensor_list([path["advantages"] for path in paths])
            env_infos = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
            agent_infos = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])

            if self._center_adv:
                advantages = util.center_advantages(advantages)

            if self._positive_adv:
                advantages = util.shift_advantages_to_positive(advantages)

            average_discounted_return = \
                np.mean([path["returns"][0] for path in paths])

            undiscounted_returns = [sum(path["rewards"]) for path in paths]

            ent = np.mean(policy.distribution.entropy(agent_infos))

            ev = special.explained_variance_1d(
                np.concatenate(baselines),
                np.concatenate(returns)
            )

            samples_data = dict(
                observations=observations,
                actions=actions,
                advantages=advantages,
                env_infos=env_infos,
                agent_infos=agent_infos,
                paths=paths,
            )
        else:
            max_path_length = max([len(path["advantages"]) for path in paths])

            # make all paths the same length (pad extra advantages with 0)
            obs = [path["observations"] for path in paths]
            obs = np.array([tensor_utils.pad_tensor(ob, max_path_length) for ob in obs])

            if self._center_adv:
                raw_adv = np.concatenate([path["advantages"] for path in paths])
                adv_mean = np.mean(raw_adv)
                adv_std = np.std(raw_adv) + 1e-8
                adv = [(path["advantages"] - adv_mean) / adv_std for path in paths]
            else:
                adv = [path["advantages"] for path in paths]

            adv = np.array([tensor_utils.pad_tensor(a, max_path_length) for a in adv])

            actions = [path["actions"] for path in paths]
            actions = np.array([tensor_utils.pad_tensor(a, max_path_length) for a in actions])

            agent_infos = [path["agent_infos"] for path in paths]
            agent_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in agent_infos]
            )

            env_infos = [path["env_infos"] for path in paths]
            env_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in env_infos]
            )

            valids = [np.ones_like(path["returns"]) for path in paths]
            valids = np.array([tensor_utils.pad_tensor(v, max_path_length) for v in valids])

            average_discounted_return = \
                np.mean([path["returns"][0] for path in paths])

            undiscounted_returns = [sum(path["rewards"]) for path in paths]

            ent = np.mean(policy.distribution.entropy(agent_infos))

            ev = special.explained_variance_1d(
                np.concatenate(baselines),
                np.concatenate(returns)
            )

            samples_data = dict(
                observations=np.asarray(obs),
                actions=np.asarray(actions),
                advantages=np.asarray(adv),
                valids=np.asarray(valids),
                agent_infos=agent_infos,
                env_infos=env_infos
            )

        logger.log("fitting baseline...")
        baseline.fit(paths)
        logger.log("fitted")

        logger.record_tabular('Iteration', itr)
        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        logger.record_tabular('AverageReturn', np.mean(undiscounted_returns))
        logger.record_tabular('ExplainedVariance', ev)
        logger.record_tabular('NumTrajs', len(paths))
        logger.record_tabular('Entropy', ent)
        logger.record_tabular('Perplexity', np.exp(ent))
        logger.record_tabular('StdReturn', np.std(undiscounted_returns))
        logger.record_tabular('MaxReturn', np.max(undiscounted_returns))
        logger.record_tabular('MinReturn', np.min(undiscounted_returns))

        return samples_data


        # samples_data["observations"] = np.asarray(obs)
        # samples_data["pdists"] = np.asarray(pdists)

        # env.log_diagnostics(env_infos)
        # policy.log_diagnostics(agent_infos)

        # env.log_extra(paths)
        # policy.log_extra(paths)
        # baseline.log_extra(paths)

        # numerical check
        # check_param = policy.get_param_values()
        # if np.any(np.isnan(check_param)):
        #     raise ArithmeticError("NaN in params")
        # elif np.any(np.isinf(check_param)):
        #     raise ArithmeticError("InF in params")

        # samples_data = dict(
        #     observations=observations,
        #     actions=actions,
        #     advantages=advantages,
        #     env_infos=env_infos,
        #     agent_infos=agent_infos,
        #     paths=paths,
        # )
        #
        # if policy.recurrent:
        #     return self.recurrent_postprocess_samples(samples_data)
        # else:
        #     return samples_data

        # def recurrent_postprocess_samples(self, samples_data):
        #     paths = samples_data["paths"]
        #
        #     max_path_length = max([len(path["advantages"]) for path in paths])
        #
        #     # make all paths the same length (pad extra advantages with 0)
        #     obs = [path["observations"] for path in paths]
        #     obs = [tensor_utils.pad_tensor(ob, max_path_length, ob[0]) for ob in obs]
        #
        #     if self._center_adv:
        #         raw_adv = np.concatenate([path["advantages"] for path in paths])
        #         adv_mean = np.mean(raw_adv)
        #         adv_std = np.std(raw_adv) + 1e-8
        #         adv = [(path["advantages"] - adv_mean) / adv_std for path in paths]
        #     else:
        #         adv = [path["advantages"] for path in paths]
        #     adv = [tensor_utils.pad_tensor(a, max_path_length, 0) for a in adv]
        #
        #     actions = [path["actions"] for path in paths]
        #     actions = [tensor_utils.pad_tensor(a, max_path_length, a[0]) for a in actions]
        #     pdists = [path["pdists"] for path in paths]
        #     pdists = [tensor_utils.pad_tensor(p, max_path_length, p[0]) for p in pdists]
        #
        #     valids = [np.ones_like(path["returns"]) for path in paths]
        #     valids = [tensor_utils.pad_tensor(v, max_path_length, 0) for v in valids]
        #
        #     samples_data["observations"] = np.asarray(obs)
        #     samples_data["advantages"] = np.asarray(adv)
        #     samples_data["actions"] = np.asarray(actions)
        #     samples_data["valids"] = np.asarray(valids)
        #     samples_data["pdists"] = np.asarray(pdists)
        #
        #     return samples_data
