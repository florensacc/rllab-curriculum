from sandbox.rocky.tf.core.network import GRUNetwork, LSTMNetwork
import sandbox.rocky.tf.core.layers as L
from enum import Enum


class NetworkType(Enum):
    GRU = "gru"
    PSEUDO_LSTM = "pseudo_lstm"
    PSEUDO_LSTM_GATE_SQUASH = "pseudo_lstm"
    TF_GRU = "tf_gru"
    TF_BASIC_LSTM = "tf_basic_lstm"
    LSTM = "lstm"
    LSTM_PEEPHOLE = "lstm_peephole"


def create_recurrent_network(network_type, **kwargs):
    if network_type == NetworkType.GRU:
        return GRUNetwork(
            gru_layer_cls=L.GRULayer,
            **kwargs
        )
    elif network_type == NetworkType.PSEUDO_LSTM:
        return LSTMNetwork(
            lstm_layer_cls=L.PseudoLSTMLayer,
            layer_args=dict(
                gate_squash_inputs=False,
            ),
            **kwargs
        )
    elif network_type == NetworkType.PSEUDO_LSTM_GATE_SQUASH:
        return LSTMNetwork(
            lstm_layer_cls=L.PseudoLSTMLayer,
            layer_args=dict(
                gate_squash_inputs=True,
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
            **kwargs
        )
    elif network_type == NetworkType.LSTM_PEEPHOLE:
        return LSTMNetwork(
            lstm_layer_cls=L.LSTMLayer,
            use_peepholes=True,
            **kwargs
        )
    else:
        raise NotImplementedError
