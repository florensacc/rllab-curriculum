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

    @property
    def support_repeat(self):
        return False

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

    @property
    def action_set(self):
        raise NotImplementedError

    @property
    def action_dims(self):
        raise NotImplementedError

    @property
    def observation_shape(self):
        raise NotImplementedError
