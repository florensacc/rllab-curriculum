import chainer
from chainer import functions as F
from chainer import links as L


class NatureDQNHead(chainer.ChainList):
    """DQN's head (Nature version)"""
    """
    outputs the hidden layer before softmax
    it's a sharable and resuable head
    """

    def __init__(self, n_input_channels=4, n_output_channels=512,
                 activation=F.relu, bias=0.1):
        # TODO: Handle different input image sizes
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_output_channels = n_output_channels

        layers = [
            L.Convolution2D(n_input_channels, 32, 8, stride=4, bias=bias),
            L.Convolution2D(32, 64, 4, stride=2, bias=bias),
            L.Convolution2D(64, 64, 3, stride=1, bias=bias),
            L.Linear(3136, n_output_channels, bias=bias),
        ]

        super(NatureDQNHead, self).__init__(*layers)

    def __call__(self, state):
        h = state
        for layer in self:
            h = self.activation(layer(h))
        return h


class NIPSDQNHead(chainer.ChainList):
    """DQN's head (NIPS workshop version)"""

    def __init__(self, n_input_channels=4, n_output_channels=256,
                 activation=F.relu, bias=0.1, img_size=84):
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_output_channels = n_output_channels

        if img_size == 44:
            fc_layer = L.Linear(512, n_output_channels, bias=bias)
        elif img_size == 84:
            fc_layer = L.Linear(2592, n_output_channels, bias=bias)
        else:
            assert False, "Can only handle two image sizes (44 or 84) currently"

        layers = [
            L.Convolution2D(n_input_channels, 16, 8, stride=4, bias=bias),
            L.Convolution2D(16, 32, 4, stride=2, bias=bias),
            fc_layer,
        ]

        super(NIPSDQNHead, self).__init__(*layers)

    def __call__(self, state):
        h = state
        for layer in self:
            h = self.activation(layer(h))
        return h
