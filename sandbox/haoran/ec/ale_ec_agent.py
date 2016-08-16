import copy
import numpy as np
import time

from rllab.misc import logger

class ALEECAgent(object):
    def __init__(
            self,
            hash_list,
            preprocessor,
            epsilon_start=1.0,
            epsilon_min=0.001,
            epsilon_decay_interval=100000,
            ucb_coeff=0.,
            discount=0.95,
            clip_reward=True,
            phi_length=4,
            testing=False,
            unpicklable_list=[],
        ):
        """
        hash_list: one hash for each action; each hash records both the max q-values and the (s,a) counts
        epsilon: decays from epsilon_start to epsilon_min over epsilon_decay_interval that many steps; then it remains constant at epsilon_min
        ucb_coeff: 0 means using epsilon-greedy exploration; positive means using ucb exploration
        phi_length: number of observations stacked to form a state (presumably a full observation)
        testing: epsilon is automatically set 0 during testing
        """
        self.hash_list = hash_list # each hash must record two attributes
        self.preprocessor = preprocessor
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay_interval = epsilon_decay_interval
        self.ucb_coeff = ucb_coeff
        self.discount = discount
        self.clip_reward = clip_reward
        self.phi_length = phi_length
        self.testing = testing
        self.unpicklable_list = unpicklable_list # just in case some large attributes should not be pickled

        self.epsilon = self.epsilon_start
        self.num_actions = len(self.hash_list)
        # set initial q-values to minus inf, so that any candidate q values will update them
        for hash in self.hash_list:
            sz = hash.tables.shape
            assert (len(sz)==3) and (sz[0] == 2)
            hash.tables[0] = -np.inf * np.ones(hash.tables.shape[1:])
            hash.tables[1] = np.zeros(hash.tables.shape[1:])
        self.total_step_count = 0

    def __getstate__(self):
        return dict(
                (k, v)
                for (k, v) in self.__dict__.iteritems()
                if k not in self.unpicklable_list
            )

    def start_episode(self, observation):
        self.cur_path = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "extra_infos": [],
        }

        self.episode_start_time = time.time()

        self.cur_path["observations"].append(observation)
        action = self.step(0,observation)

        return action


    def step(self, reward, observation):
        """
        This method is called each time step.

        Arguments:
           reward      - Real valued reward received for the previous decision.
           observation - A height x width numpy array

        Returns:
           An integer action.

        """

        self.total_step_count += 1
        self.cur_extra_info = dict()

        #TESTING---------------------------
        if self.testing:
            self.epsilon = 0
        #NOT TESTING---------------------------
        else:
            # choose a new exploration param
            self._update_epsilon()
        action = self._choose_action(observation, self.epsilon, self.ucb_coeff)


        self.cur_path["observations"].append(observation)
        self.cur_path["actions"].append(action)
        self.cur_path["rewards"].append(reward) # beware it is past reward
        self.cur_path["extra_infos"].append(
            copy.deepcopy(self.cur_extra_info)
        )

        return action


    def _update_epsilon(self):
        if self.total_step_count < self.epsilon_decay_interval:
            self.epsilon = self.epsilon_start - self.total_step_count * (self.epsilon_start - self.epsilon_min) / self.epsilon_decay_interval
        else:
            self.epsilon = self.epsilon_min


    def _choose_action(self, observation, epsilon, ucb_coeff):
        """
        Beware that some actions do not trigger the game start. So random actions are needed at the beginning.
        """
        # if not enough past observations in the current path, return a random action
        if len(self.cur_path["observations"]) < self.phi_length - 1:
            action = np.random.randint(0, self.num_actions)
            self.cur_extra_info["state_keys"] = np.zeros((self.num_actions, 1, len(self.hash_list[0].bucket_sizes)))
            self.cur_extra_info["q_values"] = np.zeros(self.num_actions)
            self.cur_extra_info["sa_counts"] = np.zeros(self.num_actions)
        else:
            # grab the last few frames in the current path and combine them with the current observation
            prev_obs = self.cur_path["observations"][-(self.phi_length-1):]
            state = np.array(prev_obs + [observation])
                # dimensions: frame, width, height
            q_values, sa_counts = self._query_state(state)
            # use epsilon-greedy for exploration
            if ucb_coeff < 1e-8:
                if np.random.uniform() < epsilon:
                    action = np.random.randint(0,self.num_actions)
                else:
                    max_actions = np.argwhere(q_values == np.amax(q_values)).ravel()
                    action = np.random.choice(max_actions)

            # use upper-confidence bound for exploration
            else:
                unexplored_actions = np.argwhere(np.isinf(q_values)).ravel()
                if len(unexplored_actions) > 0:
                    action = np.random.choice(unexplored_actions)
                else:
                    s_count = np.sum(sa_counts)
                    bonus_values = self.ucb_coeff * np.sqrt(
                        np.log(s_count) / (np.asarray(sa_counts))
                    )
                    modified_q_values = q_values + bonus_values
                    max_actions = np.argwhere(modified_q_values == np.amax(modified_q_values)).ravel()
                    action = np.random.choice(max_actions)
            print q_values, action

        return action

    def _query_state(self, state):
        state = np.reshape(state, (1,) + state.shape)
        state_processed = self.preprocessor.process(state)
        q_values = []
        sa_counts = []
        keys = []
        for hash in self.hash_list: # one hash for each action
            key = hash.compute_keys(state_processed)
            q_value = float(hash.query_keys(key, 0))
            sa_count = int(hash.query_keys(key, 1))

            q_values.append(q_value)
            sa_counts.append(sa_count)
            keys.append(key)

        # store the state key for each action hash, to avoid later re-computation
        self.cur_extra_info["state_keys"] = keys
        self.cur_extra_info["q_values"] = q_values
        self.cur_extra_info["sa_counts"] = sa_counts

        return q_values, sa_counts

    def _do_training(self):
        """
        Update all q-values along current path by episodic control
        """
        state_key_list = [
            info["state_keys"][action] for info,action in zip(self.cur_path["extra_infos"], self.cur_path["actions"])
        ]
        q_value_list = [
            info["q_values"][action] for info,action in zip(self.cur_path["extra_infos"], self.cur_path["actions"])
        ]
        count_list = [
            info["sa_counts"][action] for info,action in zip(self.cur_path["extra_infos"], self.cur_path["actions"])
        ]
        T = len(self.cur_path["actions"])
        new_q_value_list = np.zeros(T)

        rewards = self.cur_path["rewards"]
        if self.clip_reward:
            rewards = np.clip(rewards, -1, 1)
        actions = self.cur_path["actions"]
        for t in np.arange(T-1,self.phi_length-1,-1):
            hash = self.hash_list[actions[t]]

            if t == T-1:
                if rewards[T-1] != q_value_list[T-1]:
                    self.epoch_q_update_count += 1
                new_q_value_list[T-1] = rewards[T-1]
            else:
                candidate = rewards[t] + self.discount * new_q_value_list[t+1]
                # new q value from the next state is used, to accelerate information propagation
                old_q_value = q_value_list[t]
                # old q values are used, assuming the the current (s,a) does not appear later in the same path

                if candidate > old_q_value:
                    new_q_value_list[t] = candidate
                    self.epoch_q_update_count += 1
                else:
                    new_q_value_list[t] = old_q_value

            hash.set_keys(
                state_key_list[t],
                [new_q_value_list[t]],
                0,
            )
            # # if this (s,a) pair is new, directly set the q-value
            # if count_list[t] == 0:
            #     hash.set_keys(
            #         state_key_list[t],
            #         [new_q_value_list[t]],
            #         0,
            #     )
            # # otherwise, increment the q-values (more stable due to redundant buckets)
            # else:
            #     q_value_inc = new_q_value_list[t] - q_value_list[t]
            #     hash.inc_keys(
            #         state_key_list[t],
            #         [q_value_inc],
            #         0,
            #         )

            # update the (s,a) counts
            hash.inc_keys(
                state_key_list[t],
                [1],
                1,
            )

        # The stored statistics are after q and count updates
        self.epoch_q_values += list(new_q_value_list)
        self.epoch_sa_counts += list(np.array(count_list) + 1)

    def end_episode(self, reward, terminal=True):
        """
        This function is called once at the end of an episode.

        Arguments:
           reward      - Real valued reward.
           terminal    - Whether the episode ended intrinsically
                         (ie we didn't run out of steps)
        Returns:
            None
        """

        self.total_step_count += 1
        self.cur_path["rewards"].append(reward)

        if not self.testing:
            self._do_training()

        total_time = time.time() - self.episode_start_time
        episode_reward = np.sum(self.cur_path["rewards"])
        episode_length = len(self.cur_path["actions"])
        logger.log(
            """
            steps/second: {:.2f} \n
            total rewards: {:f} \n
            time steps: {:d}
            """.format(
            episode_length/total_time,
            episode_reward,
            episode_length))

    def start_epoch(self,epoch,phase):
        if phase == "Train":
            self.epoch_q_update_count = 0
            self.epoch_q_values = []
            self.epoch_sa_counts = []

    def finish_epoch(self,epoch,phase):
        if phase == "Train":
            logger.record_tabular_misc_stat(
                "StateActionCount",
                self.epoch_sa_counts,
            )
            new_sa_count = np.count_nonzero(self.epoch_sa_counts == 1)
            logger.record_tabular(
                "NewStateActionCount",
                new_sa_count,
            )
            logger.record_tabular_misc_stat(
                "QValues",
                self.epoch_q_values
            )
            logger.record_tabular(
                "QUpdate",
                self.epoch_q_update_count
            )
            logger.record_tabular(
                "Epsilon",
                self.epsilon,
            )


    def start_testing(self, epoch):
        self.testing = True

    def finish_testing(self,epoch):
        self.testing = False

    def cleanup(self):
        pass
