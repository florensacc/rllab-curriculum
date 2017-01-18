import chainer
import numpy as np
import chainer.links as L
import chainer.functions as F


class MLP(chainer.Chain):
    def __init__(
            self,
            input_shape,
            output_dim,
            hidden_sizes,
            hidden_nonlinearity,
            output_nonlinearity,
            hidden_W_init=chainer.initializers.GlorotUniform(),
            hidden_b_init=chainer.initializers.Zero(),
            output_W_init=chainer.initializers.GlorotUniform(),
            output_b_init=chainer.initializers.Zero(),
    ):
        super().__init__()
        self.l_hiddens = []
        input_dim = np.prod(input_shape)
        prev_hidden_size = input_dim
        for idx, hidden_size in enumerate(hidden_sizes):
            link = L.Linear(
                in_size=prev_hidden_size,
                out_size=hidden_size,
                initialW=hidden_W_init,
                initial_bias=hidden_b_init,
            )
            self.add_link(
                name="l_hid_{}".format(idx),
                link=link,
            )
            self.l_hiddens.append(link)
            prev_hidden_size = hidden_size
        link = L.Linear(
            in_size=prev_hidden_size,
            out_size=output_dim,
            initialW=output_W_init,
            initial_bias=output_b_init,
        )
        self.add_link(
            name="l_out",
            link=link
        )
        self.l_out = link
        if hidden_nonlinearity is None:
            hidden_nonlinearity = F.Identity()
        if output_nonlinearity is None:
            output_nonlinearity = F.Identity()
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity

    def __call__(self, x):
        for l_hid in self.l_hiddens:
            x = self.hidden_nonlinearity(l_hid(x))
        return self.output_nonlinearity(self.l_out(x))
