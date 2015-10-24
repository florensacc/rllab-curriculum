class Policy(object):

    # Should be implemented by all policies

    def get_actions(self, observations):
        raise NotImplementedError

    def get_action(self, observation):
        raise NotImplementedError

    # Only needed for parameterized policies

    def get_param_values(self):
        raise NotImplementedError

    def set_param_values(self, flattened_params):
        raise NotImplementedError

    @property
    def input_var(self):
        raise NotImplementedError

    def new_action_var(self):
        raise NotImplementedError

    # Only needed for stochastic policies

    def kl(self, old_pdist_var, new_pdist_var):
        raise NotImplementedError

    def likelihood_ratio(self, old_pdist_var, new_pdist_var, action_var):
        raise NotImplementedError

    def compute_entropy(self, pdist):
        raise NotImplementedError

    @property
    def pdist_var(self):
        raise NotImplementedError
