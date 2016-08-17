from rllab.misc import ext
from rllab.misc.overrides import overrides
from rllab.algos.batch_polopt import BatchPolopt
import rllab.misc.logger as logger
from rllab.optimizers.first_order_optimizer import FirstOrderOptimizer
from rllab.misc import special

import theano
import theano.tensor as TT
import numpy as np
import lasagne.layers as L


class ValueGradient(BatchPolopt):
    """
    Use the model to compute the value gradient.
    We can have a "natural" version as well.
    """

    def __init__(
        self,
        optimizer=None,
        optimizer_args=dict(
            batch_size=None, # no need to use minibatches
        ),
        **kwargs
    ):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = FirstOrderOptimizer(**optimizer_args)
        self.optimizer = optimizer
        super(ValueGradient,self).__init__(**kwargs)

    @overrides
    def init_opt(self):
        loss = TT.fscalar('loss')
        params = self.policy.get_params(trainable=True)
        gradients = [
            ext.new_tensor(
                name=p.name,
                ndim=len(p.get_value(borrow=True).shape),
                dtype=p.get_value(borrow=True).dtype
            )
            for p in params
        ]
        inputs = [loss] + gradients

        self.optimizer.update_opt(
            loss=loss,
            target=self.policy,
            inputs=inputs,
            gradients=gradients,
        )

    def optimize_policy(self,itr,samples_data):
        paths = samples_data["paths"]
        grads = []
        for i,path in enumerate(paths):
            logger.log("Computing value gradient of path # %d."%(i))
            V_s, V_theta = self.compute_policy_gradient(
                self.env,
                self.policy,
                path,
                self.discount,
            )
            grads.append(np.copy(V_theta[0]))
        gradient = - np.average(np.asarray(grads),axis=0) # beware this is the gradient of the loss, or minus value

        # Decompose the gradient layer-by-layer into weight matrices and bias vectors
        gradients = self.policy.flat_to_params(gradient,trainable=True)
        # param_values = L.get_all_param_values(self.policy.network.output_layer)
        # gradients = []
        # index = 0
        # for p in param_values:
        #     param_count = np.prod(p.shape)
        #     g = gradient[index : index + param_count].reshape(p.shape)
        #     # g = gradient[index : index + param_count]
        #     gradients.append(g)
        #     index += param_count

        discounted_returns = [
            special.discount_cumsum(path["rewards"], self.discount)[0]
            for path in paths
        ]

        loss = - np.average(discounted_returns)
        inputs = [loss] + gradients
        self.optimizer.optimize(inputs)

    def compute_policy_gradient(self,env,policy,path,gamma):
        T = len(path["rewards"]) # path length
        states = path["observations"]
        n = len(states[0]) # state dimension
        D = len(policy.get_theta()) # parameter dimension
        V_theta = np.zeros((T,D)) # gradient wrt params at all time steps
        V_s = np.zeros((T,n)) # gradient wrt states at all time steps

        for t in range(T-2,-1,-1):
            s = states[t]
            a = path["actions"][t]
            s2 = states[t+1]
            r_a = env.r_a(s,a)
            r_s = env.r_s(s,a)
            f_s = env.f_s(s,a)
            f_a = env.f_a(s,a)
            pi_theta = policy.pi_theta(s)
            pi_s = policy.pi_s(s)

            V_s[t] = r_s + r_a.dot(pi_s) + gamma * V_s[t+1].dot(f_s + f_a.dot(pi_s))
            V_theta[t] = r_a.dot(pi_theta) + gamma * V_s[t+1].dot(f_a.dot(pi_theta)) + gamma * V_theta[t+1]
        return V_s,V_theta

        # beware the gradients for NN are disconnected

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            env=self.env,
        )
