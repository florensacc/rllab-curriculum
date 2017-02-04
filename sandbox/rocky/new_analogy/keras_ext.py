import keras.layers as L
import keras.backend as K
import tensorflow as tf


class Identity(L.Layer):
    pass


class CausalAtrousConv1D(L.AtrousConv1D):
    def __init__(self, nb_filter, filter_length,
                 init='glorot_uniform', activation=None, weights=None,
                 border_mode='same', subsample_length=1, atrous_rate=1,
                 W_regularizer=None, b_regularizer=None,
                 activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, causal=True, **kwargs):
        assert border_mode == 'same'
        self.causal = causal
        super().__init__(nb_filter=nb_filter, filter_length=filter_length, init=init, activation=activation,
                         weights=weights, border_mode='valid', subsample_length=subsample_length,
                         atrous_rate=atrous_rate, W_regularizer=W_regularizer, b_regularizer=b_regularizer,
                         activity_regularizer=activity_regularizer, W_constraint=W_constraint,
                         b_constraint=b_constraint, bias=bias, **kwargs)

    def call(self, x, mask=None):
        k = self.filter_length
        dilation = self.atrous_rate
        padding = (k - 1) * dilation
        T = K.shape(x)[1]
        if self.causal:
            x_padded = tf.pad(x, [[0, 0], [padding, 0], [0, 0]])
        else:
            padding_left = padding // 2
            padding_right = padding - padding_left
            x_padded = tf.pad(x, [[0, 0], [padding_left, padding_right], [0, 0]])
        y_padded = super().call(x_padded, mask)
        return y_padded
        # y_sliced = y_padded[:, :T, :]
        # y_sliced.set_shape((None, None, y_padded.get_shape()[-1].value))
        # return y_sliced

    # def get_output_shape_for(self, input_shape):
    #     import ipdb; ipdb.set_trace()


def inject():
    L.Identity = Identity
    L.CausalAtrousConv1D = CausalAtrousConv1D
