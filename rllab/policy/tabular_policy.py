from rllab.policy.lasagne_policy import LasagnePolicy
from rllab.misc.serializable import Serializable
from rllab.misc.special import weighted_sample
from rllab.misc.overrides import overrides
import numpy as np
import tensorfuse as theano
import tensorfuse.tensor as TT
import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne


class TabularPolicy(LasagnePolicy, Serializable):

    def __init__(self, mdp):
        input_var = TT.matrix('input')
        l_input = L.InputLayer(
            shape=(None, mdp.observation_shape[0]),
            input_var=input_var)

        l_output = L.DenseLayer(l_input,
                                num_units=mdp.action_dim,
                                W=lasagne.init.Constant(0.),
                                b=None,
                                nonlinearity=NL.softmax)

        prob_var = L.get_output(l_output)

        self._pdist_var = prob_var
        self._prob_var = prob_var
        self._compute_probs = theano.function([input_var], prob_var,
                                              allow_input_downcast=True)
        self._input_var = input_var
        self._action_dim = mdp.action_dim
        super(TabularPolicy, self).__init__([l_output])
        Serializable.__init__(self, mdp)

    @property
    def action_dim(self):
        return self._action_dim

    @property
    @overrides
    def input_var(self):
        return self._input_var

    @property
    @overrides
    def pdist_var(self):
        return self._pdist_var

    @overrides
    def new_action_var(self, name):
        return TT.imatrix(name)

    @overrides
    def kl(self, old_prob_var, new_prob_var):
        return TT.sum(old_prob_var *
                      (TT.log(old_prob_var) - TT.log(new_prob_var)), axis=1)

    @overrides
    def likelihood_ratio(self, old_prob_var, new_prob_var, action_var):
        N = old_prob_var.shape[0]
        new_ll = new_prob_var[TT.arange(N), TT.reshape(action_var, (-1,))]
        old_ll = old_prob_var[TT.arange(N), TT.reshape(action_var, (-1,))]
        return new_ll / old_ll

    @overrides
    def compute_entropy(self, prob):
        return -np.mean(np.sum(prob * np.log(prob), axis=1))

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def get_actions(self, states):
        probs = self._compute_probs(states)
        actions = [weighted_sample(prob, range(len(prob))) for prob in probs]
        return actions, probs

    @overrides
    def get_log_prob_sym(self, action_var):
        N = action_var.shape[0]
        prob = self._prob_var[TT.arange(N), TT.reshape(action_var, (-1,))]
        return TT.log(prob)

    def get_empirical_fisher_validate(self, action_var):
        """
        Compute the symbolic empirical fisher information matrix, i.e.
        E_x [d_logp/d_theta * d_logp/d_theta.T]
        This only works for a single sample
        """
        dtheta = 0
        for i in range(self._action_dim):
            p = self.pdist_var[0, i]
            g = theano.grad(TT.log(p), self.params[0])
            dtheta += p * TT.outer(g, g)
        return dtheta


if __name__ == "__main__":
    from rllab.mdp.frozen_lake_mdp import FrozenLakeMDP
    mdp = FrozenLakeMDP(default_map='4x4')
    policy = TabularPolicy(mdp)
    policy.set_param_values(np.random.random(policy.get_param_values().shape))
    action_var = policy.new_action_var('action')
    fisher1 = policy.get_empirical_fisher(action_var)
    f_fisher1 = theano.function([policy.input_var, action_var], fisher1, on_unused_input='ignore', allow_input_downcast=True)
    inputs = np.random.choice(mdp.observation_shape[0], size=1000)
    actions = np.random.choice(mdp.action_dim, size=1000)
    inputs = np.eye(mdp.observation_shape[0])[inputs]

    emp = 0
    for idx, action in enumerate(actions):
        emp += f_fisher1(np.array([inputs[idx]]), np.array([[action]]))
    emp /= len(actions)

    another_policy = TabularPolicy(mdp)
    kl = TT.mean(policy.kl(policy.pdist_var, another_policy.pdist_var))
    fisher2 = theano.gradient.hessian(kl, policy.params)
    another_policy.set_param_values(policy.get_param_values())
    print 'compiling f fisher 2'
    f_fisher2 = theano.function([policy.input_var, action_var, another_policy.input_var], fisher2, on_unused_input='ignore', allow_input_downcast=True)
    print 'compiled'
    emp3 = f_fisher2(inputs, np.array([actions]).T, inputs)

    #kl = 
    #emp2 = 
    import ipdb; ipdb.set_trace()
