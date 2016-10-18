import time
import pickle

from sandbox.rein.algos.embedding_theano2.ale_hashing_bonus_evaluator import ALEHashingBonusEvaluator
from sandbox.rein.algos.embedding_theano2.parallel_trainer import ParallelTrainer
from sandbox.rein.algos.embedding_theano2.plotter import Plotter
from sandbox.rein.algos.embedding_theano2.trpo import TRPO
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

CONSISTENCY_CHECK_DIR = '/consistency_check'
UNIQUENESS_CHECK_DIR = '/uniqueness_check'
RANDOM_SAMPLES_DIR = '/random_samples'


class TRPOPlusLSH(TRPO):
    """
    TRPO+

    Extension to TRPO to allow for intrinsic reward.
    TRPOPlus, but with locality-sensitive hashing (LSH) on top.
    """

    def __init__(
            self,
            eta=0.1,
            model_pool_args=None,
            train_model=True,
            train_model_freq=1,
            continuous_embedding=True,
            model_embedding=True,
            sim_hash_args=None,
            clip_rewards=False,
            model_args=None,
            **kwargs):
        super(TRPOPlusLSH, self).__init__(**kwargs)

        assert model_args is not None
        assert eta >= 0
        assert train_model_freq >= 1
        if train_model:
            assert model_embedding

        self._eta = eta
        self._train_model = train_model
        self._train_model_freq = train_model_freq
        self._continuous_embedding = continuous_embedding
        self._model_embedding = model_embedding
        self._sim_hash_args = sim_hash_args
        self._clip_rewards = clip_rewards
        self._model_args = model_args

        if model_pool_args is None:
            self._model_pool_args = dict(size=100000, min_size=32, batch_size=32)
        else:
            self._model_pool_args = model_pool_args

        # start parallel trainer
        self._model_trainer = ParallelTrainer()

    def init_gpu(self):
        from sandbox.rein.dynamics_models.bnn.conv_bnn_count import ConvBNNVIME
        import theano.sandbox.cuda
        theano.sandbox.cuda.use("gpu")

        self._model = ConvBNNVIME(
            **self._model_args
        )

        if self._model_embedding:
            state_dim = self._model.discrete_emb_size
            self._hashing_evaluator_ram = ALEHashingBonusEvaluator(
                state_dim=128,
                log_prefix="ram_",
                sim_hash_args=dict(
                    dim_key=256,
                    bucket_sizes=None,
                ),
                parallel=False,
            )
        else:
            state_dim = 128

        self._hashing_evaluator = ALEHashingBonusEvaluator(
            state_dim=state_dim,
            count_target='embeddings',
            sim_hash_args=self._sim_hash_args,
        )

        if self._model_embedding:
            logger.log('Model embedding enabled.')
            if self._train_model:
                logger.log('Training model enabled.')
            else:
                logger.log('Training model disabled, using convolutional random projection.')

            if self._continuous_embedding:
                logger.log('Using continuous embedding.')
            else:
                logger.log('Using binary embedding.')
        else:
            logger.log('Model embedding disabled, using LSH directly on states.')

        if self._train_model:
            self._model_trainer.populate_trainer(pickle.dumps(self._model), self._model_pool_args)

        self._plotter = Plotter()

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

    def comp_int_rewards(self, paths):
        """
        Retrieve binary code and increase count of each sample in batch.
        :param paths:
        :return:
        """

        def bin_to_int(binary):
            integer = 0
            for bit in binary:
                integer = (integer << 1) | bit
            return integer

        def count_to_ir(count):
            return 1. / np.sqrt(count)

        def obs_to_key(path):
            if self._model_embedding:
                # Encode/decode to get uniform representation.
                obs_ed = self.decode_obs(self.encode_obs(path['env_infos']['images']))
                # Get continuous embedding.
                cont_emb = self._model.discrete_emb(obs_ed)
                if self._continuous_embedding:
                    return cont_emb
                else:
                    # Cast continuous embedding into binary one.
                    # return np.cast['int'](np.round(cont_emb))
                    bin_emb = np.cast['int'](np.round(cont_emb))
                    bin_emb_downsampled = bin_emb.reshape(-1, 8).mean(axis=1).reshape((bin_emb.shape[0], -1))
                    return np.cast['int'](np.round(bin_emb_downsampled))
            else:
                return path['observations']

        # --
        # Update counting table.
        logger.log('Retrieve embeddings ...')
        for idx, path in enumerate(paths):
            # When using num_seq_frames > 1, we need to extract the last one.
            keys = obs_to_key(path)
            path['env_infos']['embeddings'] = keys

        logger.log('Update counting table and compute intrinsic reward ...')
        self._hashing_evaluator.fit_before_process_samples(paths)
        for path in paths:
            path['S'] = self._hashing_evaluator.predict(path)

        arr_surprise = np.hstack([path['S'] for path in paths])
        logger.record_tabular('MeanS', np.mean(arr_surprise))
        logger.record_tabular('StdS', np.std(arr_surprise))

        if self._model_embedding:
            logger.log('Update counting table and compute intrinsic reward (RAM) ...')
            self._hashing_evaluator_ram.fit_before_process_samples(paths)
            for path in paths:
                path['ram_S'] = self._hashing_evaluator_ram.predict(path)
            arr_surprise_ram = np.hstack([path['ram_S'] for path in paths])
            logger.record_tabular('ram_MeanS', np.mean(arr_surprise_ram))
            logger.record_tabular('ram_StdS', np.std(arr_surprise_ram))

        logger.log('Intrinsic rewards computed')

    def fill_replay_pool(self, paths):
        """
        Fill up replay pool.
        :param paths:
        :return:
        """
        assert self._train_model
        tot_path_len = 0
        for path in paths:
            # Encode observations into replay pool format. Also make sure we only add final image in case of
            # autoencoder.
            obs_enc = self.encode_obs(path['env_infos']['images'][:, -np.prod(self._model.state_dim):])
            path_len = len(path['rewards'])
            tot_path_len += path_len
            for i in range(path_len):
                self._model_trainer.add_sample(obs_enc[i])
                # self._pool.add_sample(obs_enc[i])
        logger.log('{} samples added to replay pool'.format(tot_path_len))

    def encode_obs(self, obs):
        """
        Observation into uint8 encoding, also functions as target format
        """
        assert np.max(obs) <= 1.0
        assert np.min(obs) >= -1.0
        obs_enc = np.floor((obs + 1.0) * 0.5 * self._model.num_classes).astype("uint8")
        return obs_enc

    def decode_obs(self, obs):
        """
        From uint8 encoding to original observation format.
        """
        obs_dec = obs / float(self._model.num_classes) * 2.0 - 1.0
        assert np.max(obs_dec) <= 1.0
        assert np.min(obs_dec) >= -1.0
        return obs_dec

    def accuracy(self, _inputs, _targets):
        """
        Calculate accuracy for inputs/outputs.
        :param _inputs:
        :param _targets:
        :return:
        """
        acc = 0.
        for batch in iterate_minibatches(_inputs, _targets, 1000, shuffle=False):
            _i, _t, _ = batch
            _o = self._model.pred_fn(_i)
            _o_s = _o.reshape((-1, np.prod(self._model.state_dim), self._model.num_classes))
            _o_s = np.argmax(_o_s, axis=2)
            acc += np.sum(np.abs(_o_s - _t))
        return acc / _inputs.shape[0]

    def add_int_to_ext_rewards(self, paths):
        """
        Alter rewards in-place.
        :param paths: sampled trajectories
        :return: None
        """
        for path in paths:
            if self._clip_rewards:
                path['rewards'] = np.clip(path["ext_rewards"], -1, 1) + self._eta * path['S']
            else:
                path['rewards'] = path["ext_rewards"] + self._eta * path['S']

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

        # --
        # Observations are concatenations of RAM and img.
        # Actual observation = RAM, split off img into 'img'
        for path in paths:
            path['images'] = np.array(path['observations'][128:])

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
            rnd = np.random.randint(0, len(paths[0]['env_infos']['images']), 32)
            self._test_obs = self.encode_obs(
                paths[0]['env_infos']['images'][rnd, -np.prod(self._model.state_dim):])
        obs = self.encode_obs(paths[0]['env_infos']['images'][-32:, -np.prod(self._model.state_dim):])

        # inputs = np.random.randint(0, 2, (10, 128))
        # self._plotter.plot_gen_imgs(
        #     model=self._model, inputs=inputs, targets=self._test_obs,
        #     itr=0, dir='/generated')

        if self._train_model and itr % 30 == 0:
            logger.log('Plotting random samples ...')
            batch = self._model_trainer._pool.random_batch(32)
            _x = self.decode_obs(batch['observations'])
            _y = batch['observations']
            self._plotter.plot_pred_imgs(model=self._model, inputs=_x, targets=_y, itr=0,
                                         dir=RANDOM_SAMPLES_DIR)
            self._plotter.print_embs(model=self._model, counting_table=None, inputs=_x,
                                     dir=RANDOM_SAMPLES_DIR, hamming_distance=0)

        if self._model_embedding and self._train_model and itr % 30 == 0:
            logger.log('Plotting consistency images ...')
            self._plotter.plot_pred_imgs(
                model=self._model, inputs=self.decode_obs(self._test_obs), targets=self._test_obs,
                itr=-itr - 1, dir=CONSISTENCY_CHECK_DIR)
            logger.log('Plotting uniqueness images ...')
            self._plotter.plot_pred_imgs(
                model=self._model, inputs=self.decode_obs(obs), targets=obs, itr=0,
                dir=UNIQUENESS_CHECK_DIR)

        if self._model_embedding:
            logger.log('Printing embeddings ...')
            self._plotter.print_consistency_embs(
                model=self._model, counting_table=None, inputs=self.decode_obs(self._test_obs),
                dir=CONSISTENCY_CHECK_DIR, hamming_distance=0)
            self._plotter.print_embs(
                model=self._model, counting_table=None, inputs=self.decode_obs(obs),
                dir=UNIQUENESS_CHECK_DIR, hamming_distance=0)

        # --
        # Diagnostics
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

        self.init_gpu()

        start_time = time.time()
        for itr in range(self.n_itr):
            with logger.prefix('itr #%d | ' % itr):
                # --
                # Sample trajectories.
                paths = self.obtain_samples(itr)

                # --
                # Preprocess trajectory data.
                self.preprocess(paths)

                if self._train_model:
                    # --
                    # Fill replay pool.
                    self.fill_replay_pool(paths)

                    # First iteration, train sequentially.
                    if itr == 0:
                        self._model_trainer.train_model()
                        params = self._model_trainer.output_q.get()
                        self._model.set_param_values(params)

                    if itr != 0 and itr % self._train_model_freq == 0:
                        if itr != self._train_model_freq:
                            params = self._model_trainer.output_q.get()
                            self._model.set_param_values(params)

                        self._model_trainer.train_model()

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
        self._model_trainer.terminate()
