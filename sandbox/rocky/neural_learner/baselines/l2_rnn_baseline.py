from rllab.core.serializable import Serializable
from rllab.misc import logger
from sandbox.rocky.neural_learner.optimizers.tbptt_optimizer import TBPTTOptimizer
from sandbox.rocky.tf.baselines.base import Baseline
from sandbox.rocky.tf.core.layers_powered import LayersPowered
import tensorflow as tf
import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.misc import tensor_utils

from sandbox.rocky.tf.policies.rnn_utils import NetworkType, create_recurrent_network
import numpy as np


# Performs least-squares regression. No trust region involved
class L2RNNBaseline(Baseline, LayersPowered, Serializable):
    def __init__(
            self,
            name,
            env_spec,
            hidden_dim=32,
            feature_network=None,
            state_include_action=True,
            hidden_nonlinearity=tf.tanh,
            network_type=NetworkType.GRU,
            weight_normalization=False,
            layer_normalization=False,
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
            super(L2RNNBaseline, self).__init__(env_spec)

            obs_dim = env_spec.observation_space.flat_dim
            action_dim = env_spec.action_space.flat_dim

            if state_include_action:
                input_dim = obs_dim + action_dim
            else:
                input_dim = obs_dim

            l_input = L.InputLayer(
                shape=(None, None, input_dim),
                name="input"
            )

            if feature_network is None:
                feature_dim = input_dim
                l_flat_feature = None
                l_feature = l_input
            else:
                feature_dim = feature_network.output_layer.output_shape[-1]
                l_flat_feature = feature_network.output_layer
                l_feature = L.OpLayer(
                    l_flat_feature,
                    extras=[l_input],
                    name="reshape_feature",
                    op=lambda flat_feature, input: tf.reshape(
                        flat_feature,
                        tf.pack([tf.shape(input)[0], tf.shape(input)[1], feature_dim])
                    ),
                    shape_op=lambda _, input_shape: (input_shape[0], input_shape[1], feature_dim)
                )

            prediction_network = create_recurrent_network(
                network_type,
                input_shape=(feature_dim,),
                input_layer=l_feature,
                output_dim=1,
                hidden_dim=hidden_dim,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=None,
                weight_normalization=weight_normalization,
                layer_normalization=layer_normalization,
                name="prediction_network",
            )

            self.prediction_network = prediction_network
            self.feature_network = feature_network
            self.l_input = l_input
            self.state_include_action = state_include_action

            flat_input_var = tf.placeholder(dtype=tf.float32, shape=(None, input_dim), name="flat_input")
            if feature_network is None:
                feature_var = flat_input_var
            else:
                feature_var = L.get_output(l_flat_feature, {feature_network.input_layer: flat_input_var})

            self.input_dim = input_dim
            self.action_dim = action_dim
            self.hidden_dim = hidden_dim
            self.state_dim = prediction_network.state_dim
            self.batch_size = batch_size
            self.n_steps = n_steps

            self.prev_actions = None
            self.prev_states = None

            out_layers = [prediction_network.output_layer]
            if feature_network is not None:
                out_layers.append(feature_network.output_layer)

        if optimizer is None:
            optimizer = TBPTTOptimizer()

        self.optimizer = optimizer
        self.log_loss_before = log_loss_before
        self.log_loss_after = log_loss_after
        self.moments_update_rate = moments_update_rate

        state_input_var = tf.placeholder(tf.float32, (None, prediction_network.state_dim), "state")
        recurrent_state_output = dict()

        predict_flat_input_var = tf.reshape(
            l_input.input_var,
            tf.pack((
                tf.shape(l_input.input_var)[0] *
                tf.shape(l_input.input_var)[1],
                tf.shape(l_input.input_var)[2]
            ))
        )

        prediction_var = L.get_output(
            prediction_network.output_layer,
            {feature_network.input_layer: predict_flat_input_var},
            recurrent_state={prediction_network.recurrent_layer: state_input_var},
            recurrent_state_output=recurrent_state_output,
        )
        direct_prediction_var = L.get_output(
            prediction_network.output_layer,
            {feature_network.input_layer: predict_flat_input_var},
        )

        state_output = recurrent_state_output[prediction_network.recurrent_layer]
        final_state = tf.reverse(state_output, [False, True, False])[:, 0, :]

        return_var = tf.placeholder(dtype=tf.float32, shape=(None, None), name="return")
        valid_var = tf.placeholder(dtype=tf.float32, shape=(None, None), name="valid")

        return_mean_var = tf.Variable(
            np.cast['float32'](0.),
            # np.zeros((), dtype=np.float32),
            name="return_mean",
        )
        return_std_var = tf.Variable(
            np.cast['float32'](1.),
            # np.ones((), dtype=np.float32),
            name="return_std",
        )

        normalized_return_var = (return_var - return_mean_var) / return_std_var

        residue = tf.reshape(prediction_var, (-1,)) - tf.reshape(normalized_return_var, (-1,))

        loss_var = tf.reduce_sum(tf.square(residue) * tf.reshape(valid_var, (-1,))) / tf.reduce_sum(valid_var)

        self.f_predict = tensor_utils.compile_function(
            inputs=[l_input.input_var],
            outputs=direct_prediction_var * return_std_var + return_mean_var,
        )
        self.f_predict_stateful = tensor_utils.compile_function(
            inputs=[l_input.input_var, state_input_var],
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
        LayersPowered.__init__(self, out_layers)

        self.optimizer.update_opt(
            loss=loss_var,
            target=self,
            inputs=[l_input.input_var, return_var, valid_var],
            rnn_state_input=state_input_var,
            rnn_final_state=final_state,
            rnn_init_state=prediction_network.state_init_param,
        )

    def predict(self, path):
        obs = path["observations"]
        if self.state_include_action:
            raise NotImplementedError
        else:
            all_input = [obs]
        return self.f_predict(all_input).flatten()

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
            batch_paths = sorted_paths[batch_idx:batch_idx+batch_size]
            batch_T = max([len(p["rewards"]) for _, p in batch_paths])
            batch_obs = tensor_utils.pad_tensor_n([p["observations"] for _, p in batch_paths], batch_T)
            states = np.tile(
                self.prediction_network.state_init_param.eval().reshape((1, -1)),
                (len(batch_obs), 1)
            )
            for t in range(0, batch_T, n_steps):
                time_sliced_obs = batch_obs[:, t:t+n_steps]
                batch_results, states = self.f_predict_stateful(time_sliced_obs, states)
                all_results[batch_idx:batch_idx+len(batch_obs), t:t+time_sliced_obs.shape[1]] = batch_results[:, :, 0]
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
