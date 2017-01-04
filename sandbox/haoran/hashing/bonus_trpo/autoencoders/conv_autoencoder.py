class SpatialAutoencoder(object):
    def __init__(
        self,
        encoder_params,
        decoder_params,
        hidden_W_init=LI.GlorotUniform(), hidden_b_init=LI.Constant(0.),
        output_W_init=LI.GlorotUniform(), output_b_init=LI.Constant(0.),
        hidden_nonlinearity=LN.rectify,
        output_nonlinearity=None,
        name=None,
    ):
    """
    :param encoder_params, decoder_params: has
    """
