import numpy as np

import sandbox.rocky.tf.core.layers as L
from rllab.misc import logger
from sandbox.rocky.tf.core.layers_powered import LayersPowered
from sandbox.rocky.tf.policies import rnn_utils
from sandbox.rocky.tf.distributions.bernoulli import Bernoulli
import tensorflow as tf
from sandbox.rocky.tf.misc import tensor_utils


class RNNPredictionBonusEvaluator(LayersPowered):
    def __init__(
            self,
            env_spec,
            hidden_dim=50,
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.sigmoid,
            no_improvement_tolerance=5,
            n_epochs=20,
            n_bprop_steps=None,
            train_ratio=0.7,
            learning_rate=1e-2,
            # store a maximum of 500 paths
            max_pool_size=500,
    ):
        """
        We can maintain a pool of past samples
        """
        self.env_spec = env_spec
        self.observation_space = env_spec.observation_space
        self.action_space = env_spec.action_space
        self.obs_dim = env_spec.observation_space.flat_dim
        self.action_dim = env_spec.action_space.flat_dim
        self.hidden_dim = hidden_dim
        self.n_bprop_steps = n_bprop_steps

        self.raw_obs_dim = env_spec.observation_space.components[0].flat_dim

        self.no_improvement_tolerance = no_improvement_tolerance
        self.n_epochs = n_epochs
        self.train_ratio = train_ratio
        self.learning_rate = learning_rate
        self.max_pool_size = max_pool_size

        self.network = rnn_utils.create_recurrent_network(
            network_type=rnn_utils.NetworkType.GRU,
            weight_normalization=True,
            layer_normalization=False,
            name="predictor",
            input_shape=(self.raw_obs_dim + 1 + self.action_dim,),
            output_dim=self.raw_obs_dim,
            hidden_dim=hidden_dim,
            hidden_nonlinearity=hidden_nonlinearity,
            # Each individual unit is either on or off?
            output_nonlinearity=output_nonlinearity,
        )

        self.pool = None

        LayersPowered.__init__(self, self.network.output_layer)
        self.f_train = None
        self.init_opt()

        self.error_mean = 0.
        self.error_std = 1.

    def init_opt(self):
        obs_term_var = tf.placeholder(tf.float32, (None, None, self.raw_obs_dim + 1), "obs_term")
        next_obs_var = tf.placeholder(tf.float32, (None, None, self.raw_obs_dim), "next_obs")
        action_var = tf.placeholder(tf.float32, (None, None, self.action_dim), "action")
        valid_var = tf.placeholder(tf.float32, (None, None), name="valid")

        input_var = tf.concat(2, [obs_term_var, action_var])

        state_var = tf.placeholder(tf.float32, (None, self.network.state_dim), "state")

        recurrent_state_output = dict()
        train_predicted_next_obs = L.get_output(
            self.network.output_layer,
            inputs={self.network.input_layer: input_var},
            recurrent_state_output=recurrent_state_output,
            recurrent_state={self.network.recurrent_layer: state_var}
        )
        predicted_next_obs = L.get_output(
            self.network.output_layer,
            inputs={self.network.input_layer: input_var},
        )

        state_output = recurrent_state_output[self.network.recurrent_layer]

        final_state = tf.reverse(state_output, [False, True, False])[:, 0, :]

        train_loss = tf.reduce_sum(
            Bernoulli(self.raw_obs_dim).kl_sym(dict(p=next_obs_var), dict(p=train_predicted_next_obs)) * valid_var
        ) / tf.reduce_sum(valid_var)
        loss = tf.reduce_sum(
            Bernoulli(self.raw_obs_dim).kl_sym(dict(p=next_obs_var), dict(p=predicted_next_obs)) * valid_var
        ) / tf.reduce_sum(valid_var)

        predict_error = tf.reduce_mean(tf.square(next_obs_var - predicted_next_obs), reduction_indices=-1)

        lr_var = tf.placeholder(tf.float32, (), name="lr")

        params = self.get_params(trainable=True)

        train_op = tf.train.AdamOptimizer(learning_rate=lr_var).minimize(train_loss, var_list=params)

        self.f_train = tensor_utils.compile_function(
            inputs=[obs_term_var, action_var, next_obs_var, valid_var, state_var, lr_var],
            outputs=[train_op, train_loss, final_state]
        )

        self.f_loss = tensor_utils.compile_function(
            inputs=[obs_term_var, action_var, next_obs_var, valid_var],
            outputs=loss,
        )

        self.f_predict_error = tensor_utils.compile_function(
            inputs=[obs_term_var, action_var, next_obs_var],
            outputs=predict_error,
        )

    def predict(self, path):
        raw_obs = path["observations"][:-1, :self.raw_obs_dim]
        terms = path["observations"][1:, -1:]
        raw_obs_term = np.concatenate([raw_obs, terms], axis=-1)
        raw_next_obs = path["observations"][1:, :self.raw_obs_dim]
        actions = path["actions"][:-1]
        errors = self.f_predict_error([raw_obs_term], [actions], [raw_next_obs])
        return np.append(errors[0], 0)

    def fit_before_process_samples(self, paths):
        # re-normalize all predictions
        # errors = np.concatenate(list(map(self.predict, paths)), axis=0)
        # self.error_mean = np.mean(errors)
        # self.error_std = np.std(errors) + 1e-5
        pass

    def fit_after_process_samples(self, samples_data):
        # paths = samples_data["paths"]
        # We have 10 trajectories
        raw_obs = samples_data["observations"][:, :-1, :self.raw_obs_dim]
        # Note that we shift the index of terminal signals by 1, since we need to know whether the current episode is
        #  finished when making predictions about the next observation
        terms = samples_data["observations"][:, 1:, -1:]
        raw_obs_term = np.concatenate([raw_obs, terms], axis=-1)
        raw_next_obs = samples_data["observations"][:, 1:, :self.raw_obs_dim]
        actions = samples_data["actions"][:, :-1]
        valids = samples_data["valids"][:, :-1]

        learning_rate = self.learning_rate

        best_train_loss = None
        best_val_loss = None
        best_train_params = None
        best_val_params = None
        n_no_train_improvement = 0
        n_no_val_improvement = 0

        # Perform truncated back prop

        sess = tf.get_default_session()

        new_data = [raw_obs_term, actions, raw_next_obs, valids]
        # should pad all to the same length
        if self.pool is None:
            self.pool = new_data
        else:
            if len(raw_obs_term[0]) != len(self.pool[0][0]):
                # need to pad one of them
                max_len = max(len(raw_obs_term[0]), len(self.pool[0][0]))
                if len(raw_obs_term[0]) > len(self.pool[0][0]):
                    # need to pad existing items in the pool
                    new_data = [
                        np.asarray([tensor_utils.pad_tensor(x, max_len) for x in data_item])
                        for data_item in new_data
                    ]
                else:
                    self.pool = [
                        np.asarray([tensor_utils.pad_tensor(x, max_len) for x in data_item])
                        for data_item in self.pool
                    ]

            self.pool = [np.concatenate([x, y], axis=0)[-self.max_pool_size:] for x, y in zip(self.pool, new_data)]

        data_list = self.pool
        n_data = len(data_list[0])
        n_train = int(np.floor(n_data * self.train_ratio))
        n_val = n_data - n_train
        train_data = [x[:n_train] for x in data_list]
        val_data = [x[n_train:] for x in data_list]

        logger.log("#data: %d" % n_data)

        # if False:

        if self.n_bprop_steps is None:
            n_bprop_steps = raw_obs_term.shape[1]
        else:
            n_bprop_steps = self.n_bprop_steps

        for epoch_idx in range(self.n_epochs):
            state = np.tile(
                sess.run(self.network.state_init_param).reshape((1, -1)),
                (n_train, 1)
            )

            train_losses = []
            for i in range(0, raw_obs_term.shape[1], n_bprop_steps):
                batch_data = [x[:, i:i + n_bprop_steps] for x in train_data]
                _, loss, state = self.f_train(*(batch_data + [state, learning_rate]))
                train_losses.append(loss)
            state = np.tile(
                sess.run(self.network.state_init_param).reshape((1, -1)),
                (n_val, 1)
            )
            train_loss = np.mean(train_losses)
            # Evaluate val loss
            val_loss = self.f_loss(*(val_data + [state]))
            logger.log("Epoch: %3d\t Train Loss: %f\t Val Loss: %f" % (epoch_idx, np.mean(train_losses), val_loss))

            if best_train_loss is None or train_loss < best_train_loss:
                best_train_loss = train_loss
                best_train_params = self.get_param_values()
                n_no_train_improvement = 0
            else:
                n_no_train_improvement += 1

            if n_no_train_improvement >= self.no_improvement_tolerance:
                logger.log("No improvement for %d epochs. Reducing learning rate to %f" % (
                    n_no_train_improvement, learning_rate))
                learning_rate *= 0.5
                n_no_train_improvement = 0
                self.set_param_values(best_train_params)

            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_params = self.get_param_values()
                n_no_val_improvement = 0
            else:
                n_no_val_improvement += 1

            if n_no_val_improvement >= self.no_improvement_tolerance:
                logger.log(
                    "Terminating early since no improvement for validation error for %d epochs" % n_no_val_improvement)
                break

        logger.log("Best val loss: %f" % best_val_loss)

        logger.record_tabular("Bonus_ValLoss", best_val_loss)
        logger.record_tabular("Bonus_NEpochs", epoch_idx)

        self.set_param_values(best_val_params)

    def log_diagnostics(self, paths):
        pass
