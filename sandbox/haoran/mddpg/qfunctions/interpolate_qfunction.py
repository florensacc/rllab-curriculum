from rllab.core.serializable import Serializable
import tensorflow as tf
import numpy as np

class InterpolateQFunction(Serializable):
    def __init__(
        self,
        scope_name,
        discrete_Q,
        env_spec,
        s_grid_sizes,
        a_grid_sizes,
        action_input=None,
        reuse=False,
    ):
        """
        :param discrete_Q: an (ns, na) matrix
        """
        Serializable.quick_init(self, locals())
        self.scope_name = scope_name
        self.discrete_Q = discrete_Q.astype(np.float32)
        self.env_spec = env_spec
        self.s_grid_sizes = s_grid_sizes
        self.a_grid_sizes = a_grid_sizes

        s_dim = env_spec.observation_space.flat_dim
        s_bounds = env_spec.observation_space.bounds
        a_dim = env_spec.action_space.flat_dim
        a_bounds = env_spec.action_space.bounds

        self.interpolator = LinearInterpolator()

        with tf.variable_scope(self.scope_name, reuse=reuse) as variable_scope:
            if action_input is None:
                self.actions_placeholder = tf.placeholder(
                    tf.float32,
                    shape=[None, a_dim],
                    name='critic_actions',
                )
            else:
                self.actions_placeholder = action_input
            self.observations_placeholder = tf.placeholder(
                tf.float32,
                shape=[None, s_dim],
                name='critic_observations',
            )
            self.variable_scope = variable_scope

            s_normalized = self.normalize(
                self.observations_placeholder, s_bounds, s_grid_sizes)
            a_normalized = self.normalize(
                self.actions_placeholder, a_bounds, a_grid_sizes)

            coordinates = tf.concat(
                1, [s_normalized, a_normalized])
            self._output = self.interpolator.interpolate(
                self.discrete_Q, coordinates)

            self.useless_internal_param = tf.constant(1.)

        self._sess = None


    def normalize(self, coordinates, bounds, grid_sizes):
        """
        :param coordinates: (None, n)
        :param bounds = (lbs, ubs), each a length n vector
        :param grid_sizes: a length n vector

        :return similar to pixel indices of the coordinates, as in an image
        """
        lbs, ubs = bounds
        n = range(len(lbs))

        coordinates_clipped = tf.maximum(
            tf.minimum(
                coordinates,
                tf.cast(tf.expand_dims(ubs, 0), tf.float32),
            ),
            tf.cast(tf.expand_dims(lbs, 0), tf.float32),
        )

        coordinates_normalized = tf.pack([
            (coordinates_clipped[:,i] - lb) / grid_size
            for i, lb, grid_size in zip(n, lbs, grid_sizes)
        ], axis=1)
        return coordinates_normalized

    def get_copy(self, **kwargs):
        return Serializable.clone(
            self,
            **kwargs
        )

    def get_weight_tied_copy(self, action_input):
        return self.__class__(
            scope_name=self.scope_name,
            discrete_Q=self.discrete_Q,
            env_spec=self.env_spec,
            s_grid_sizes=self.s_grid_sizes,
            a_grid_sizes=self.a_grid_sizes,
            action_input=action_input,
            reuse=True,
        )

    @property
    def output(self):
        return self._output

    def get_params_internal(self, **kwargs):
        return [self.useless_internal_param]

    def get_param_values(self):
        return None

    def set_param_values(self, params):
        pass

    @property
    def sess(self):
        if self._sess is None:
            self._sess = tf.get_default_session()
        return self._sess

    @sess.setter
    def sess(self, value):
        self._sess = value

class LinearInterpolator(object):
    def __init__(self):
        pass

    def interpolate(self, values, coordinates):
        """
        outputs = \sum_{b \in {0,1}^k} w * v[c_b],
            where c_b[i] = floor(c_b[i]) if b[i] == 0, else ceil(c_b[i]),
            and w = \prod_i (c_b[i] - floor) if b[i] == 1 else (ceil - c_b[i])

        :param values: a multi-dim array of shape (n1, n2, ..., nk)
        :param coordinates: a set of coordinates of shape (N, k),
            each coordinate in the range [0, ni-1]
        :return: a vector of length N of linearly interpolated values
        """
        k = len(values.shape)
        offsets = [np.prod(values.shape[i+1:]).astype(np.int32) for i in range(k)]

        floors, ceils, weights = [], [], []
        for i in range(k):
            C = coordinates[:,i]
            floors.append(tf.cast(tf.floor(C), tf.int32))
            ceils.append(tf.cast(tf.ceil(C), tf.int32))
            weights.append(C - tf.floor(C)) # OK

        outputs = tf.zeros_like(coordinates[:,0])
        values_flat = tf.reshape(values, (-1,))
        formatter = "0:0%db"%(k)
        corner_types = [("{" + formatter + "}").format(i) for i in range(2 ** k)]

        for corner_type in corner_types:
            corner_coordinates = tf.pack([
                floors[i] if corner_type[i] == "0" else ceils[i]
                for i in range(k)
            ], axis=1)
            corner_indices = tf.zeros_like(outputs, dtype=tf.int32)
            for i in range(k):
                corner_indices = corner_indices + \
                    tf.cast(offsets[i] * corner_coordinates[:,i], tf.int32)

            corner_values = tf.gather(values_flat, corner_indices)

            W = tf.ones_like(outputs)
            for i in range(k):
                if corner_type[i] == "0":
                    w = 1. - weights[i]
                else:
                    w = weights[i]
                W = W * w
            outputs += W * corner_values

        return outputs


    def get_corner_indices(self, coordinates, corner_type):
        indices = []
        for i, b in enumerate(corner_type):
            if b == "0":
                corners = tf.floor(coordinates[:,i])
            elif b == "1":
                corners = tf.ceil(coordinates[:,i])
            indices.append(tf.cast(corners, tf.int32))
        return tf.pack(indices, axis=1)

import joblib
class DataLoader(object):
    def __init__(self, file, key):
        self.file = file
        self.key = key

    def load(self):
        return joblib.load(self.file)[self.key]
