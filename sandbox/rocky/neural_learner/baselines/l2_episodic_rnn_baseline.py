from rllab.core.serializable import Serializable
from rllab.misc import logger
from sandbox.rocky.analogy.rnn_cells import GRUCell
from sandbox.rocky.neural_learner.episodic.episodic_cell import EpisodicCell
from sandbox.rocky.neural_learner.episodic.episodic_network_builder import EpisodicNetworkBuilder
from sandbox.rocky.neural_learner.optimizers.tbptt_optimizer import TBPTTOptimizer
from sandbox.rocky.tf.baselines.base import Baseline
from sandbox.rocky.tf.core.layers_powered import LayersPowered
import tensorflow as tf
from sandbox.rocky.tf.misc import tensor_utils
import sandbox.rocky.tf.core.layers as L

from sandbox.rocky.tf.policies.rnn_utils import NetworkType, create_recurrent_network
import numpy as np

from sandbox.rocky.tf.spaces import Box
from sandbox.rocky.tf.spaces import Product, Discrete


# Performs least-squares regression. No trust region involved
class L2EpisodicRNNBaseline(Baseline, LayersPowered, Serializable):
    def __init__(
            self,
            name,
            env_spec,
            cell=None,
            network_builder=None,
            optimizer=None,
            # these are only used when computing predictions in batch
            batch_size=None,
            n_steps=None,
            log_loss_before=True,
            log_loss_after=True,
            moments_update_rate=0.9,
    ):
        Serializable.quick_init(self, locals())
        """
        :param env_spec: A spec for the env.
        :param hidden_dim: dimension of hidden layer
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :return:
        """

        self.observation_space = env_spec.observation_space
        self.action_space = env_spec.action_space

        with tf.variable_scope(name):
            Baseline.__init__(self, env_spec)

            obs_dim = env_spec.observation_space.flat_dim
            l_obs_input = L.InputLayer(
                shape=(None, None, obs_dim),
                name="obs"
            )

            if network_builder is None:
                network_builder = EpisodicNetworkBuilder(env_spec)

            if cell is None:
                cell = EpisodicCell(GRUCell(num_units=128, activation=tf.nn.relu, weight_normalization=True))

            l_obs_components = network_builder.split_obs_layer(l_obs_input)
            l_raw_obs, l_prev_action, l_reward, l_terminal = l_obs_components
            l_obs_feature = network_builder.new_obs_feature_layer(l_raw_obs)
            l_action_feature = network_builder.new_action_feature_layer(l_prev_action)
            self.obs_feature_dim = l_obs_feature.output_shape[-1]
            self.action_feature_dim = l_action_feature.output_shape[-1]

            l_feature = L.concat([l_obs_feature, l_action_feature, l_reward, l_terminal], axis=2)

            l_rnn = network_builder.new_rnn_layer(
                l_feature,
                cell=cell,
            )
            l_v = L.TemporalUnflattenLayer(
                L.DenseLayer(
                    L.TemporalFlattenLayer(l_rnn),
                    num_units=1,
                    nonlinearity=None,
                ),
                ref_layer=l_obs_input
            )

            self.l_obs_input = l_obs_input
            self.l_obs_components = l_obs_components
            self.l_feature = l_feature
            self.l_rnn = l_rnn

            self.batch_size = batch_size
            self.n_steps = n_steps

        if optimizer is None:
            optimizer = TBPTTOptimizer()

        self.optimizer = optimizer
        self.log_loss_before = log_loss_before
        self.log_loss_after = log_loss_after
        self.moments_update_rate = moments_update_rate

        state_input_var = tf.placeholder(tf.float32, (None, l_rnn.state_dim), "state")
        recurrent_state_output = dict()

        prediction_var = L.get_output(
            l_v,
            recurrent_state={l_rnn: state_input_var},
            recurrent_state_output=recurrent_state_output,
        )
        direct_prediction_var = L.get_output(
            l_v,
        )

        final_state = recurrent_state_output[l_rnn]

        return_var = tf.placeholder(dtype=tf.float32, shape=(None, None), name="return")
        valid_var = tf.placeholder(dtype=tf.float32, shape=(None, None), name="valid")

        return_mean_var = tf.Variable(
            np.cast['float32'](0.),
            name="return_mean",
        )
        return_std_var = tf.Variable(
            np.cast['float32'](1.),
            name="return_std",
        )

        normalized_return_var = (return_var - return_mean_var) / return_std_var

        residue = tf.reshape(prediction_var, (-1,)) - tf.reshape(normalized_return_var, (-1,))

        loss_var = tf.reduce_sum(tf.square(residue) * tf.reshape(valid_var, (-1,))) / tf.reduce_sum(valid_var)

        self.f_predict = tensor_utils.compile_function(
            inputs=[l_obs_input.input_var],
            outputs=direct_prediction_var * return_std_var + return_mean_var,
        )
        self.f_predict_stateful = tensor_utils.compile_function(
            inputs=[l_obs_input.input_var, state_input_var],
            outputs=[prediction_var * return_std_var + return_mean_var, final_state],
        )

        return_mean_stats = tf.reduce_sum(return_var * valid_var) / tf.reduce_sum(valid_var)
        return_std_stats = tf.sqrt(
            tf.reduce_sum(tf.square(return_var - return_mean_var) * valid_var) / tf.reduce_sum(valid_var)
        )

        self.f_update_stats = tensor_utils.compile_function(
            inputs=[return_var, valid_var],
            outputs=[
                tf.assign(
                    return_mean_var,
                    (1 - self.moments_update_rate) * return_mean_var + \
                    self.moments_update_rate * return_mean_stats,
                ),
                tf.assign(
                    return_std_var,
                    (1 - self.moments_update_rate) * return_std_var + \
                    self.moments_update_rate * return_std_stats,
                )
            ]
        )

        self.return_mean_var = return_mean_var
        self.return_std_var = return_std_var
        LayersPowered.__init__(self, l_v)

        self.optimizer.update_opt(
            loss=loss_var,
            target=self,
            inputs=[l_obs_input.input_var, return_var, valid_var],
            rnn_state_input=state_input_var,
            rnn_final_state=final_state,
            rnn_init_state=tf.reshape(l_rnn.cell.zero_state(1, dtype=tf.float32), (-1,)),
        )

    def predict(self, path):
        obs = path["observations"]
        return self.f_predict([obs]).flatten()

    def predict_n(self, paths):
        N = len(paths)
        T = max([len(p["rewards"]) for p in paths])
        if self.n_steps is None:
            n_steps = T
        else:
            n_steps = self.n_steps
        if self.batch_size is None:
            batch_size = N
        else:
            batch_size = self.batch_size
        # sort paths by their lengths, so that we don't mix short and long trajs together
        sorted_paths = sorted(enumerate(paths), key=lambda x: len(x[1]["rewards"]))
        all_results = np.zeros((N, T))
        for batch_idx in range(0, N, batch_size):
            batch_paths = sorted_paths[batch_idx:batch_idx + batch_size]
            batch_T = max([len(p["rewards"]) for _, p in batch_paths])
            batch_obs = tensor_utils.pad_tensor_n([p["observations"] for _, p in batch_paths], batch_T)
            states = np.zeros((len(batch_obs), self.l_rnn.state_dim))
            for t in range(0, batch_T, n_steps):
                time_sliced_obs = batch_obs[:, t:t + n_steps]
                batch_results, states = self.f_predict_stateful(time_sliced_obs, states)
                all_results[batch_idx:batch_idx + len(batch_obs), t:t + time_sliced_obs.shape[1]] = batch_results[:, :,
                                                                                                    0]
        ordered_results = [all_results[idx] for idx, _ in sorted_paths]
        return [ordered_results[idx][:len(path["rewards"])].flatten() for idx, path in enumerate(paths)]

    def get_params_internal(self, **tags):
        params = LayersPowered.get_params_internal(self, **tags)
        if not tags.get('trainable', False):
            params = params + [self.return_mean_var, self.return_std_var]
        return params

    def get_param_values(self, **tags):
        return LayersPowered.get_param_values(self, **tags)

    def set_param_values(self, flattened_params, **tags):
        LayersPowered.set_param_values(self, flattened_params, **tags)

    def fit_with_samples(self, paths, samples_data):
        inputs = [samples_data["observations"], samples_data["returns"], samples_data["valids"]]

        self.f_update_stats(samples_data["returns"], samples_data["valids"])

        with logger.prefix("Vf | "), logger.tabular_prefix("Vf."):
            if self.log_loss_before:
                logger.log("Computing loss before training")
                loss_before, _ = self.optimizer.loss_diagnostics(inputs)
                logger.log("Computed")

            epoch_losses = []

            def record_data(loss, diagnostics, *args, **kwargs):
                epoch_losses.append(loss)
                return True

            self.optimizer.optimize(inputs, callback=record_data)

            if self.log_loss_after:
                logger.log("Computing loss after training")
                loss_after, _ = self.optimizer.loss_diagnostics(inputs)
                logger.log("Computed")

            # perform minibatch gradient descent on the surrogate loss, while monitoring the KL divergence

            if self.log_loss_before:
                logger.record_tabular('LossBefore', loss_before)
            else:
                # Log approximately
                logger.record_tabular('FirstEpoch.Loss', epoch_losses[0])
            if self.log_loss_after:
                logger.record_tabular('LossAfter', loss_after)
            else:
                logger.record_tabular('LastEpoch.Loss', epoch_losses[-1])
            if self.log_loss_before and self.log_loss_after:
                logger.record_tabular('dLoss', loss_before - loss_after)
