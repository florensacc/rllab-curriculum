"""
IDEA:
1. When updating the resetter, choose the path with highest return. Find the last state with non-zero rewards and reset to that state deterministically.
2. We assume that infinite time is allowed to complete the game and rewards are undiscounted.
3. Because internal states can not be transferred between different environment instances, we need to perform master actions on each environment instance to reach that "good" state, and then use the environment's internal saveState() function to remember it.
4. Potential problem: the resulting trajectory can be very long.
5. Is there an extreme case in which the agent dies right after getting the last reward? Or how can we justify this approach by categorizing all suitable MDPs?
6. How does the agent benefit from previous sub-episodes? The hash count or the policy / Q network.
"""
import numpy as np
import os
import cv2
import copy
from sandbox.haoran.myscripts.myutilities import get_time_stamp
from sandbox.haoran.hashing.bonus_trpo.envs.atari_env import AtariEnv
from rllab.misc import logger


class AtariSaveLoadResetter(object):
    def __init__(self,
            restored_state_folder="/tmp/restored_state",
            avoid_life_lost=False,
        ):
        self.master_actions = np.array([]) # stub does not like a list
        self.new_actions = None
        self.restored_state_folder = restored_state_folder
        self.avoid_life_lost = avoid_life_lost
        self.updated_image_output = True
        self.init_reward = 0
        self.init_state = None
        if restored_state_folder is not None:
            if not os.path.isdir(restored_state_folder):
                os.system("mkdir -p %s"%(restored_state_folder))

    def update(self,_paths):
        self.updated_image_output = True
        # Make a copy of _paths (collected for training) so that we don't mess up with the optimizer. Copy any information you need, except large ones like images.
        paths = [
            dict(
                raw_rewards=p["raw_rewards"],
                actions=p["actions"],
                env_infos=dict(
                    lives_lost=p["env_infos"]["lives_lost"]
                )
            )
            for p in _paths
        ]

        # remove prior_reward (the reward collected by master actions)
        for path in paths:
            path["raw_rewards"][0] = 0

        # only keep the part of traj before lossing lives
        if self.avoid_life_lost:
            for path in paths:
                life_lost_times = np.where(
                    path["env_infos"]["lives_lost"] == True
                )[0]
                if len(life_lost_times) > 0:
                    # Life is lost after taking a_{t1} at s_{t1}; so in principle a different action could have been taken to avoid life lost. Though for Atari don't hold the hope.
                    t1 = life_lost_times[0] + 1
                    # the code below can be written more hierarchically
                    for key in path.keys():
                        if key not in ["env_infos","agent_infos"]: # not dicts
                            path[key] = path[key][:t1]
                        else:
                            for sub_key in path[key].keys():
                                path[key][sub_key] = path[key][sub_key][:t1]

        returns = np.asarray([np.sum(path["raw_rewards"]) for path in paths])
        if np.amax(returns) == 0:
            logger.log("Unable to discover new rewards in the current epoch.",
                color="red",
            )
            self.new_actions = None
            logger.record_tabular("ResetterNewRewards", 0)
        else:
            # find out paths with highest return
            highest_return_paths = [
                paths[i] for i in range(len(returns))
                if returns[i] == np.amax(returns)
            ]
            final_reward_times = np.asarray([
                np.where(path["raw_rewards"] > 0)[0][-1]
                for path in highest_return_paths
            ])
            # best path: highest raw rewards within shortest time
            best_path_index = np.where(final_reward_times==np.amin(final_reward_times))[0][0]
            best_path = highest_return_paths[best_path_index]

            # path["actions"] are one-hot; need to convert them to integers
            best_path_actions = [
                np.where(a==1)[0][0]
                for a in best_path["actions"]
            ]
            self.new_actions = list(best_path_actions[:np.amin(final_reward_times)+1])
            self.master_actions = np.asarray(list(self.master_actions) + self.new_actions)

            new_rewards = np.sum(best_path["raw_rewards"])
            logger.log("Discovered new actions \n %s\n to total rewards %d!"%\
                (self.new_actions,new_rewards),
                color="green",
            )
            logger.record_tabular("ResetterNewRewards", new_rewards)
            self.updated_image_output = False
        logger.log("Current master actions: %s"%(self.master_actions))

    def reset(self,env):
        assert isinstance(env,AtariEnv)
        assert env.max_start_nullops == 0 # cannot deal with stochastic env

        use_default_reset = (self.init_state is None) and (self.new_actions is None)

        # restore to the last "good state"
        if self.init_state is None:
            env.ale.reset_game()
            logger.log("Restored to default initial state")
        else:
            env.ale.reset_game()
            env.ale.restoreSystemState(self.init_state)
            logger.log("Restored to state %s"%(self.init_state))

        # After each iteration, the central resetter will update new actions and broadcast them to worker resetters. While sampling the first path, the worker resetters take new actions to reach the next good state, and recrods that state for easier reset during later sampling.
        if self.new_actions is not None:
            for a in self.new_actions:
                o, r, done, env_info = env.step(a)
                self.init_reward += r
            self.new_actions = None
            self.init_state = env.ale.cloneSystemState()
            logger.log("Incorporated new actions and cloned the new good state.")
        env._prior_reward = self.init_reward

        # save the restored state for debugging
        # only one worker will print; that's what you need
        if len(self.master_actions) > 0 and \
            self.restored_state_folder is not None and \
            not self.updated_image_output:

            if env.game_name == "breakout":
                env.ale.act(0) # forces the image to show up
                img = env.ale.getScreenRGB()[:,:,::-1]
                env.ale.restoreSystemState(self.init_state)
            else:
                img = env.ale.getScreenRGB()[:,:,::-1]

            timestamp = get_time_stamp()
            filename = os.path.join(self.restored_state_folder,timestamp+".jpg")
            cv2.imwrite(filename, img)
            logger.log("Snapthost restored state to %s"%(filename))
            self.updated_image_output = True

        return use_default_reset

    def get_param_values(self):
        params = dict(
            master_actions=self.master_actions,
            new_actions=self.new_actions,
            updated_image_output=self.updated_image_output,
        )
        return params

    def set_param_values(self,params):
        self.master_actions = params["master_actions"]
        self.new_actions = params["new_actions"]
        self.updated_image_output = params["updated_image_output"]
