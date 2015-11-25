from rllab.misc import autoargs


class Policy(object):

    # Should be implemented by all policies

    def get_actions(self, observations):
        raise NotImplementedError

    def get_action(self, observation):
        actions, pdists = self.get_actions([observation])
        return actions[0], pdists[0]

    # Only needed for parameterized policies

    def get_param_values(self):
        raise NotImplementedError

    def set_param_values(self, flattened_params):
        raise NotImplementedError

    @property
    def input_var(self):
        raise NotImplementedError

    def new_action_var(self, name):
        raise NotImplementedError

    # Only needed for stochastic policies

    def kl(self, old_pdist_var, new_pdist_var):
        raise NotImplementedError

    def likelihood_ratio(self, old_pdist_var, new_pdist_var, action_var):
        raise NotImplementedError

    def compute_entropy(self, pdist):
        raise NotImplementedError

    # Only needed for vanilla policy gradient & guided policy search
    def get_log_prob(self, observations, actions):
        if not hasattr(self, '_f_log_prob'):
            action_var = self.new_action_var()
            self._f_log_prob = theano.function(
                [self.input_var, action_var],
                self.get_lob_prob_sym(action_var),
                allow_input_downcast=True,
                on_unused_input='ignore'
            )
        return self._f_log_prob(observations, actions)

    def get_log_prob_sym(self, action_var):
        raise NotImplementedError

    @property
    def pdist_var(self):
        raise NotImplementedError

    @classmethod
    @autoargs.add_args
    def add_args(cls, parser):
        pass

    @classmethod
    @autoargs.new_from_args
    def new_from_args(cls, args, mdp):
        pass
