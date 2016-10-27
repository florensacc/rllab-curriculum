import time

from sandbox.rein.algos.ganx_th.gan_bonus_evaluator import GANBonusEvaluator
from sandbox.rein.algos.ganx_th.trpo import TRPO
from rllab.misc import special
import numpy as np
from rllab.misc import tensor_utils
import rllab.misc.logger as logger
from rllab.algos import util
from sandbox.rein.dynamics_models.utils import iterate_minibatches

# --
# Nonscientific printing of numpy arrays.
np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)


class TRPOPlus(TRPO):
    """
    TRPO+

    Extension to TRPO to allow for intrinsic reward.
    """

    def __init__(
            self,
            eta=0.1,
            bonus_evaluator_args=None,
            **kwargs):
        super(TRPOPlus, self).__init__(**kwargs)

        assert eta >= 0

        # Exploration bonus factor.
        self._eta = eta

        if bonus_evaluator_args is None:
            model_pool_args = dict(
                size=100000,
                min_size=32,
                batch_size=32
            )
            bonus_evaluator_args = dict(
                observation_shape=(self.env.observation_space.flat_dim,),
                observation_dtype="uint8",
                model_pool_args=model_pool_args
            )
        self._bonus_evaluator = GANBonusEvaluator(**bonus_evaluator_args)

    def process_samples(self, itr, paths):
        baselines = []
        returns = []
        for path in paths:
            path_baselines = np.append(self.baseline.predict(path), 0)
            deltas = path["rewards"] + \
                     self.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            path["advantages"] = special.discount_cumsum(
                deltas, self.discount * self.gae_lambda)
            path["returns"] = special.discount_cumsum(path["rewards"], self.discount)
            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])

        ev = special.explained_variance_1d(
            np.concatenate(baselines),
            np.concatenate(returns)
        )

        if not self.policy.recurrent:
            observations = tensor_utils.concat_tensor_list([path["observations"] for path in paths])
            actions = tensor_utils.concat_tensor_list([path["actions"] for path in paths])
            rewards = tensor_utils.concat_tensor_list([path["rewards"] for path in paths])
            returns = tensor_utils.concat_tensor_list([path["returns"] for path in paths])
            advantages = tensor_utils.concat_tensor_list([path["advantages"] for path in paths])
            env_infos = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
            agent_infos = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])

            if self.center_adv:
                advantages = util.center_advantages(advantages)

            if self.positive_adv:
                advantages = util.shift_advantages_to_positive(advantages)

            average_discounted_return = \
                np.mean([path["returns"][0] for path in paths])

            undiscounted_returns = [sum(path["ext_rewards"]) for path in paths]

            ent = np.mean(self.policy.distribution.entropy(agent_infos))

            samples_data = dict(
                observations=observations,
                actions=actions,
                rewards=rewards,
                returns=returns,
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

            if self.center_adv:
                raw_adv = np.concatenate([path["advantages"] for path in paths])
                adv_mean = np.mean(raw_adv)
                adv_std = np.std(raw_adv) + 1e-8
                adv = [(path["advantages"] - adv_mean) / adv_std for path in paths]
            else:
                adv = [path["advantages"] for path in paths]

            adv = np.array([tensor_utils.pad_tensor(a, max_path_length) for a in adv])

            actions = [path["actions"] for path in paths]
            actions = np.array([tensor_utils.pad_tensor(a, max_path_length) for a in actions])

            rewards = [path["rewards"] for path in paths]
            rewards = np.array([tensor_utils.pad_tensor(r, max_path_length) for r in rewards])

            returns = [path["returns"] for path in paths]
            returns = np.array([tensor_utils.pad_tensor(r, max_path_length) for r in returns])

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

            undiscounted_returns = [sum(path["ext_rewards"]) for path in paths]

            ent = np.sum(self.policy.distribution.entropy(agent_infos) * valids) / np.sum(valids)

            samples_data = dict(
                observations=obs,
                actions=actions,
                advantages=adv,
                rewards=rewards,
                returns=returns,
                valids=valids,
                agent_infos=agent_infos,
                env_infos=env_infos,
                paths=paths,
            )

        logger.log("Updating baseline ...")
        self.baseline.fit(paths)
        logger.log("Baseline updated.")

        logger.record_tabular('Iteration', itr)
        logger.record_tabular('AverageDiscountedReturn', average_discounted_return)
        logger.record_tabular_misc_stat('AverageReturn', undiscounted_returns)
        logger.record_tabular('ExplainedVariance', ev)
        logger.record_tabular('NumTrajs', len(paths))
        logger.record_tabular('Entropy', ent)
        logger.record_tabular('Perplexity', np.exp(ent))

        return samples_data

    def add_int_to_ext_rewards(self, paths):
        """
        Alter rewards in-place.
        :param paths: sampled trajectories
        :return: None
        """
        for path in paths:
            path['S'] = self._bonus_evaluator.compute_bonus(path)
            path['rewards'] += self._eta * path['S']

    @staticmethod
    def preprocess(paths):
        """
        Preprocess data.
        :param paths:
        :return:
        """
        # --
        # Save external rewards.
        for path in paths:
            path['ext_rewards'] = np.array(path['rewards'])

    def diagnostics(self, start_time, itr, samples_data, paths):
        """
        Diagnostics of each run.
        :param start_time:
        :param itr:
        :param samples_data:
        :param paths:
        """
        # --
        # Analysis
        # Get consistency images in first iteration.
        if itr == 0:
            # Select random images form the first path, evaluate them at every iteration to inspect emb.
            rnd = np.random.randint(0, len(paths[0]['observations']), 32)
            self._test_obs = self.encode_obs(paths[0]['observations'][rnd, -np.prod(self._model.state_dim):])
        obs = self.encode_obs(paths[0]['observations'][-32:, -np.prod(self._model.state_dim):])

        # --
        # Diagnositic
        self.log_diagnostics(paths)
        logger.log("Saving snapshot ...")
        params = self.get_itr_snapshot(itr, samples_data)  # , **kwargs)
        if self.store_paths:
            params["paths"] = samples_data["paths"]
        logger.save_itr_params(itr, params)
        logger.log("saved")
        logger.record_tabular('Time', time.time() - start_time)
        logger.dump_tabular(with_prefix=False)
        if self.plot:
            self.update_plot()
            if self.pause_for_plot:
                input("Plotting evaluation run: Press Enter to continue...")

    def train(self):
        """
        Main RL training procedure.
        """
        self.start_worker()
        self.init_opt()
        start_time = time.time()
        for itr in range(self.n_itr):
            with logger.prefix('itr #%d | ' % itr):
                # --
                # Sample trajectories.
                paths = self.obtain_samples(itr)

                # --
                # Preprocess trajectory data.
                self.preprocess(paths)

                self._bonus_evaluator.before_process_samples(paths)
                self._bonus_evaluator.train_model(itr)

                # --
                # Compute intrinisc rewards.
                self.comp_int_rewards(paths)

                # --
                # Add intrinsic reward to external: 'rewards' is what is actually used as 'true' reward.
                self.add_int_to_ext_rewards(paths)

                # --
                # Compute deltas, advantages, etc.
                samples_data = self.process_samples(itr, paths)

                # --
                # Optimize policy according to latest trajectory batch `samples_data`.
                self.optimize_policy(itr, samples_data)

                # --
                # Diagnosics
                self.diagnostics(start_time, itr, samples_data, paths)

        self.shutdown_worker()
