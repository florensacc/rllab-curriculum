from rllab.baselines.base import Baseline
from rllab.misc.overrides import overrides
from rllab.regressors.gaussian_mlp_regressor import GaussianMLPRegressor
from rllab.optimizers.lbfgs_optimizer import LbfgsOptimizer
from rllab.misc import logger
import lasagne.nonlinearities as NL

import numpy as np
from scipy.linalg import hankel


class ExtremeModelBaseline(Baseline):
    def __init__(
            self,
            env_spec,
            lookahead=0,
            batch_size=4000,
            discount=1,
            experience_limit=None,
            max_opt_itr=20,
            nonlinearity=NL.rectify,
            network_size=(32, 32),
    ):
        self.lookahead = lookahead
        self.env_spec = env_spec
        self.experience = None
        self.batch_size = batch_size
        self.discount = discount
        self.experience_limit = experience_limit
        self.nonlinearity = nonlinearity
        self.max_opt_itr = max_opt_itr
        self.network_size = network_size

        self.experience = []

        state_dimensionality = env_spec.observation_space.flat_dim
        action_dimensionality = env_spec.action_space.flat_dim

        self.value_network = GaussianMLPRegressor(
            input_shape=(state_dimensionality + 1,),
            output_dim=1,
            learn_std=False,
            name="vf",
            use_trust_region=False,
            optimizer=LbfgsOptimizer(max_opt_itr=max_opt_itr),
            hidden_sizes=network_size,
            hidden_nonlinearity=nonlinearity
            )

        self.dynamics_network = GaussianMLPRegressor(
            input_shape=(state_dimensionality + action_dimensionality,),
            output_dim=state_dimensionality,
            name="dn",
            use_trust_region=False,
            optimizer=LbfgsOptimizer(max_opt_itr=max_opt_itr),
            hidden_sizes=network_size,
            hidden_nonlinearity=NL.tanh,
            )

        self.rewards_network = GaussianMLPRegressor(
            input_shape=(state_dimensionality + action_dimensionality,),
            output_dim=1,
            learn_std=False,
            name="rn",
            use_trust_region=False,
            optimizer=LbfgsOptimizer(max_opt_itr=max_opt_itr),
            hidden_sizes=network_size,
            hidden_nonlinearity=NL.tanh,
            )

        # self.debug_mlp_baseline = GaussianMLPBaseline(env_spec=env_spec) Send Rocky error message

        # Use name argument
        # Use more optimizer iterations (maybe 1000)
        # Check equivalence of Gaussian MLP Value estimators, use same initialization
        # Change nonlinearity to tanh
        # Hidden size (200, 200)
        # Compare network sizes, optimization iterations, nonlinearity

    @overrides
    def get_param_values(self, **tags):
        return NotImplementedError

    @overrides
    def set_param_values(self, val, **tags):
        return NotImplementedError

    @overrides
    def extreme(self):
        return True

    @overrides
    def fit(self, paths):
        # Fit value estimator; depends on policy, so can only use most recent batch
        path_length, _ = paths[0]['observations'].shape  # For now, assumes uniform path length (e.g. Half Cheetah)
        timesteps = np.arange(path_length).reshape(-1, 1) / float(path_length)
        states_with_timestep = np.concatenate(
            [np.concatenate([path['observations'], timesteps], axis=1) for path in paths])
        returns = np.concatenate([path['returns'] for path in paths], axis=0).reshape(-1, 1)

        self.value_network.fit(states_with_timestep, returns)

        # Dynamics and reward don't depend on policy, so we can use old information
        self.experience.extend(paths)
        if self.experience_limit is not None \
                and len(self.experience) > self.experience_limit * len(paths):
            self.experience = self.experience[len(paths):]

        logger.record_tabular("NumExperiences", len(self.experience))

        # Fit dynamics model
        all_states = np.concatenate([path['observations'] for path in self.experience], axis=0)
        all_actions = np.concatenate([path['actions'] for path in self.experience], axis=0)
        state_action_pairs = np.concatenate([all_states, all_actions], axis=1)
        successor_states = all_states[1:]

        self.dynamics_network.fit(state_action_pairs[:-1], successor_states)

        # Fit reward model
        rewards = np.concatenate([path['rewards'] for path in self.experience], axis=0).reshape(-1, 1)

        self.rewards_network.fit(state_action_pairs, rewards)

    def predict_unvectorized(self, path, policy):
        """Returns expected discounted return according to learned model and fixed noise.
        Uses a value estimator after self.timesteps steps.
        """
        noise = path['agent_infos']['noise']
        states = path['observations']
        path_length, state_dimensionality = states.shape
        num_paths = max(1, self.batch_size / path_length)  # Must have at least one path
        baselines = []  # baselines[i] is estimate of V(s_i) (zero-indexed)
        for t, start_state in enumerate(states):
            batch = []
            for _ in range(num_paths):
                state = start_state
                empirical_return = 0
                action = policy.get_action(state)
                for i in range(self.lookahead):
                    if t+i >= path_length:  # We have reached the end of the path; no more reward
                        break
                    if i > 0:  # We don't fix noise for the first time step
                        action = policy.get_action_with_fixed_noise(state, noise[t+i])
                    state_action_pair = np.concatenate([state, action], axis=1)
                    reward = self.rewards_network.predict([state_action_pair])[0]
                    next_state = self.dynamics_network.sample_predict([state_action_pair])[0]
                    empirical_return += reward
                    state = next_state
                if t+i < path_length:  # If there is any future remaining
                    empirical_return += self.value_network.predict([np.append(state, t+i)])
                batch.append(empirical_return)
            baselines.append(float(sum(batch)) / len(batch))
        return baselines

    def predict_partially_vectorized(self, path, policy, unvectorized=False):
        if unvectorized:
            return self.predict_unvectorized(path, policy)  # For debugging purposes

        noise = path['agent_infos']['noise']
        states_along_path = path['observations']
        path_length, _ = states_along_path.shape
        num_paths = max(1, self.batch_size / path_length)  # Must have at least one path
        baselines = []
        for t in range(path_length):
            rollout_states = np.repeat(states_along_path[t], num_paths, axis=0)
            returns = np.zeros((num_paths,))
            actions = policy.get_actions(rollout_states)
            for i in range(self.lookahead):
                if t+i >= path_length:
                    break
                if i > 0:  # Don't fix noise for the first time step
                    actions = policy.get_actions_with_fixed_noise(rollout_states, noise[t+i])
                state_action_pairs = np.concatenate(rollout_states, actions, axis=1)
                returns += self.discount**i * self.rewards_network.predict(state_action_pairs)
                rollout_states = self.dynamics_network.sample_predict(state_action_pairs)
            if t+i < path_length:
                timestep = np.repeat(t+i, num_paths).reshape(-1, 1)
                final_states_with_timestep = np.concatenate([rollout_states, timestep], axis=1)
                returns += self.discount**i * self.value_network.predict(final_states_with_timestep)
            baselines.append(np.average(returns))
        return baselines

    @overrides
    def predict(self, path, policy):
        states_along_path = path['observations']
        path_length, _ = states_along_path.shape
        num_paths = max(1, self.batch_size / path_length)  # Must have at least one path
        noise = self._cascading_noise_tensor(path['agent_infos']['noise']).repeat(num_paths, axis=1)
        rollout_states = states_along_path.repeat(num_paths, axis=0)
        returns = np.zeros((num_paths * path_length),)
        actions, _ = policy.get_actions(rollout_states)

        i = 0  # For handling the self.lookahead = 0 case
        for i in range(min(self.lookahead, path_length)):
            if i > 0:  # Don't fix noise for first time step
                # import pdb; pdb.set_trace()
                actions, _ = policy.get_actions_with_fixed_noise(rollout_states, noise[i])
            state_action_pairs = np.concatenate([rollout_states, actions], axis=1)
            rewards = self.discount**i * self.rewards_network.predict(state_action_pairs)
            if i > 0:  # Unfortunately if i == 0 this zeros out the whole array
                rewards[-num_paths * i:] = 0
            returns += rewards.flatten()
            rollout_states = self.dynamics_network.sample_predict(state_action_pairs)

        timestep = np.arange(i, i + path_length).repeat(num_paths).reshape(-1, 1) / float(path_length)
        final_states_with_timestep = np.concatenate([rollout_states, timestep], axis=1)
        value_estimates = self.discount**i * self.value_network.predict(final_states_with_timestep)
        if i > 0:  # If i == 0 this zeros out the whole array
            value_estimates[-num_paths * i:] = 0
        returns += value_estimates.flatten()

        baselines = np.average(returns.reshape(path_length, num_paths), axis=1)
        return baselines

    def _cascading_noise_tensor(self, noise_matrix):
        path_length, action_dim = noise_matrix.shape
        return hankel(noise_matrix)[::action_dim].reshape(path_length, path_length, action_dim)
