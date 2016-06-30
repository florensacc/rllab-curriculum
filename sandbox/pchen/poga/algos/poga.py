from rllab.core.network import MLP
from rllab.misc import ext
from rllab.misc.ext import compile_function
from rllab.misc.overrides import overrides
from rllab.algos.batch_polopt import BatchPolopt
import rllab.misc.logger as logger
import theano
import theano.tensor as TT

from rllab.optimizers.first_order_optimizer import FirstOrderOptimizer
from rllab.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from sandbox.rocky.tf.misc import tensor_utils
import numpy as np


class POGA(BatchPolopt):

    def __init__(
            self,
            inner_algo,
            optimizer=FirstOrderOptimizer(
                max_epochs=1,
                batch_size=None,
                learning_rate=1e-2,
            ),
            **kwargs
    ):
        self._inner = inner_algo
        self._optimizer = optimizer
        super(POGA, self).__init__(**kwargs)

    @overrides
    def init_opt(self):
        self._inner.init_opt()
        import lasagne.nonlinearities as NL

        obs_var = self.env.observation_space.new_tensor_variable('obs',extra_dims=1)
        # import ipdb; ipdb.set_trace()
        disc_net = MLP(
            input_shape=self.env.observation_space.shape,
            input_var=obs_var,
            output_dim=1,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=NL.leaky_rectify,
            output_nonlinearity=NL.sigmoid,
        )
        tgt_var = ext.new_tensor(
            'discrim tgt',
            ndim=1,
            dtype=theano.config.floatX
        )
        import lasagne.objectives as OB
        disc_out = TT.reshape(disc_net.output, (-1,))
        disc_loss = TT.mean(OB.binary_crossentropy(disc_out, tgt_var))
        input_lst = [
            obs_var,
            tgt_var,
        ]
        self._optimizer.update_opt(
            loss=disc_loss,
            target=disc_net,
            inputs=input_lst,
        )

        self._disc_scorer = compile_function([obs_var], -TT.log(disc_out))

    @overrides
    def optimize_policy(self, itr, samples_data):
        paths = samples_data["paths"]
        sorted_paths = sorted(paths, key=lambda path: -path["returns"][0])
        cur_obs = samples_data["observations"]
        cur_tgt = np.zeros_like(cur_obs[:, 0])

        better_paths = sorted_paths[:int(0.2*len(paths))]
        better_obs = tensor_utils.concat_tensor_list([path["observations"] for path in better_paths])
        better_tgt = np.ones_like(better_obs[:, 0])

        # XXX unbalanced sample size
        fin = ([
            np.concatenate([cur_obs, better_obs], axis=0),
            np.concatenate([cur_tgt, better_tgt], axis=0),
        ])

        self._optimizer.optimize(fin)
        for _ in xrange(1):
            for path in paths:
                path["rewards"] = -self._disc_scorer(*[
                    path["observations"],
                ])
            ga_samples_data = self.sampler.process_samples(itr, paths)
            self._inner.optimize_policy(itr, ga_samples_data)
            paths = self.sampler.obtain_samples(itr)

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
