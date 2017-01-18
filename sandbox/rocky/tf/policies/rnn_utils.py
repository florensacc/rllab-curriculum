from sandbox.rocky.tf.core.network import GRUNetwork, LSTMNetwork, TfRNNNetwork
import sandbox.rocky.tf.core.layers as L
from enum import Enum


class NetworkType(Enum):
    GRU = "gru"
    PSEUDO_LSTM = "pseudo_lstm"
    PSEUDO_LSTM_GATE_SQUASH = "pseudo_lstm_gate_squash"
    TF_GRU = "tf_gru"
    TF_BASIC_LSTM = "tf_basic_lstm"
    LSTM = "lstm"
    LSTM_PEEPHOLE = "lstm_peephole"
    TF_RNN = "tf_rnn"


def create_recurrent_network(network_type, W_x_init=L.XavierUniformInitializer(), W_h_init=L.OrthogonalInitializer(),
                             layer_normalization=False,
                             weight_normalization=False,
                             **kwargs):
    """
    Allows creating rnns with string-like specifications. Example:
    """
    if network_type == NetworkType.GRU:
        return GRUNetwork(
            gru_layer_cls=L.GRULayer,
            layer_args=dict(
                W_x_init=W_x_init,
                W_h_init=W_h_init,
                layer_normalization=layer_normalization,
                weight_normalization=weight_normalization,
            ),
            **kwargs
        )
    elif network_type == NetworkType.PSEUDO_LSTM:
        return LSTMNetwork(
            lstm_layer_cls=L.PseudoLSTMLayer,
            layer_args=dict(
                gate_squash_inputs=False,
                W_x_init=W_x_init,
                W_h_init=W_h_init,
                layer_normalization=layer_normalization,
                weight_normalization=weight_normalization,
            ),
            **kwargs
        )
    elif network_type == NetworkType.PSEUDO_LSTM_GATE_SQUASH:
        return LSTMNetwork(
            lstm_layer_cls=L.PseudoLSTMLayer,
            layer_args=dict(
                gate_squash_inputs=True,
                W_x_init=W_x_init,
                W_h_init=W_h_init,
                layer_normalization=layer_normalization,
                weight_normalization=weight_normalization,
            ),
            **kwargs
        )
    elif network_type == NetworkType.TF_GRU:
        return GRUNetwork(
            gru_layer_cls=L.TfGRULayer,
            **kwargs
        )
    elif network_type == NetworkType.TF_BASIC_LSTM:
        return LSTMNetwork(
            lstm_layer_cls=L.TfBasicLSTMLayer,
            use_peepholes=False,
            **kwargs
        )
    elif network_type == NetworkType.LSTM:
        return LSTMNetwork(
            lstm_layer_cls=L.LSTMLayer,
            use_peepholes=False,
            layer_args=dict(
                W_x_init=W_x_init,
                W_h_init=W_h_init,
                layer_normalization=layer_normalization,
                weight_normalization=weight_normalization,
            ),
            **kwargs
        )
    elif network_type == NetworkType.LSTM_PEEPHOLE:
        return LSTMNetwork(
            lstm_layer_cls=L.LSTMLayer,
            use_peepholes=True,
            layer_args=dict(
                W_x_init=W_x_init,
                W_h_init=W_h_init,
                layer_normalization=layer_normalization,
                weight_normalization=weight_normalization,
            ),
            **kwargs
        )
    elif network_type == NetworkType.TF_RNN:
        return TfRNNNetwork(
            **kwargs
        )
    else:
        raise NotImplementedError(network_type)
