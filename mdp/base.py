
class States(object):
    """
    Stores a collection of MDP states
    """
    def copy(self):
        "optional -- copy the state"
        raise NotImplementedError


class MDP(object):
    def step(self, states, actions):
        """        
        s,a -> s', o, r, d

        Inputs
        ------
        states
        actions

        Returns
        -------
        (nextstates, observation, rewards, done)
        """
        raise NotImplementedError

    def step_single(self, state, action):
        next_states, obs, rewards, dones, effective_steps = self.step([state], map(lambda x: [x], action))
        return next_states[0], obs[0], rewards[0], dones[0], effective_steps[0]

    def sample_initial_state(self):
        states, obs = self.sample_initial_states(1)
        return states[0], obs[0]

    def sample_initial_states(self, n):
        """
        Sample n initial states. Also return initial observations

        Inputs
        ------
        n : integer number of states, n >= 1

        Returns
        -------
        s : instance of States
        o : observation (tuple of ndarray)
        """
        raise NotImplementedError
    def action_spec(self):
        """
        Return either a pair (dtype, shape)
        Or a list of triples (fieldname, dtype, shape)
        """
        raise NotImplementedError
    def observation_spec(self):
        """
        Return either a pair (dtype, shape)
        Or a list of triples (fieldname, dtype, shape)
        """
        raise NotImplementedError        
    def reward_names(self):
        """
        Return tuple of names of reward terms
        """
        raise NotImplementedError
    def plot(self, states, actions=None):
        """
        Plot states and actions.
        Should accept actions=None
        """
        raise NotImplementedError
    def text(self, states):
        """
        Return textual representation of state
        """
        raise NotImplementedError

class Policy(object):
    def step(self, o):
        """
        Return dict including

        required: 
            a : actions
        optional:
            pa : specifies probability distribution that 'a' was sampled from
            [whatever else your learning algorithm will need]
        """
        raise NotImplementedError
    def output_spec(self):
        """
        returns a dict name->spec providing a specification of the outputs of step()
        each 'spec' is a list of triples (fieldname, dtype, shape)
        """
        raise NotImplementedError


