from sandbox.rocky.tf.core.network import GRUNetwork, LSTMNetwork
import sandbox.rocky.tf.core.layers as L


def create_recurrent_network(network_type, **kwargs):
    if network_type == "gru":
        return GRUNetwork(
            gru_layer_cls=L.GRULayer,
            **kwargs
        )
    elif network_type == "pseudo_lstm":
        return GRUNetwork(
            gru_layer_cls=L.PseudoLSTMLayer,
            **kwargs
        )
    elif network_type == "tf_gru":
        return GRUNetwork(
            gru_layer_cls=L.TfGRULayer,
            **kwargs
        )
    elif network_type == "tf_basic_lstm":
        return LSTMNetwork(
            lstm_layer_cls=L.TfBasicLSTMLayer,
            **kwargs
        )
    elif network_type == "lstm":
        return LSTMNetwork(
            lstm_layer_cls=L.LSTMLayer,
            **kwargs
        )
    elif network_type == "lstm_peephole":
        return LSTMNetwork(
            lstm_layer_cls=L.LSTMLayer,
            use_peepholes=True,
            **kwargs
        )
    else:
        raise NotImplementedError