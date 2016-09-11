"""
IDEA:
1. When updating the resetter, choose the path with highest return. Find the last state with non-zero rewards and reset to that state deterministically.
2. We assume that infinite time is allowed to complete the game.
3. Because internal states can not be transferred between different environment instances, we need to remember the entire action sequence leading to that 'good' state.
4. Potential problem: the resulting trajectory can be very long. But from the computer's point of view, repeating the same action sequence may not be a big deal.
5. Is there an extreme case in which the agent dies right after getting the last reward?
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
            restored_state_folder="/tmp/restored_state"
        ):
        self.master_actions = np.array([])
        self.restored_state_folder = restored_state_folder
        if restored_state_folder is not None:
            if not os.path.isdir(restored_state_folder):
                os.system("mkdir -p %s"%(restored_state_folder))

    def update(self,paths):
        # remove the first part of each path given by the master action
        t0 = len(self.master_actions)
        paths = copy.deepcopy(paths) # avoid modifying the paths for policy optimization
        for path in paths:
            for key in path.keys():
                if key not in ["env_infos","agent_infos"]: # hack
                    path[key] = path[key][t0:]
                else:
                    for sub_key in path[key].keys():
                        path[key][sub_key] = path[key][sub_key][t0:]

        returns = np.asarray([np.sum(path["raw_rewards"]) for path in paths])
        if np.amax(returns) == 0:
            logger.log("Unable to discover new rewards in the current epoch.",
                color="red",
            )
            use_default_reset = True
        else:
            # find out paths with highest return
            highest_return_paths = [
                paths[i] for i in range(len(paths))
                if returns[i] == np.amax(returns)
            ]
            final_reward_times = np.asarray([
                np.where(path["raw_rewards"] > 0)[0][-1]
                for path in highest_return_paths
            ])
            best_path_index = np.where(final_reward_times==np.amin(final_reward_times))[0][0]
            best_path = highest_return_paths[best_path_index]
            best_path_actions = [
                np.where(a==1)[0][0]
                for a in best_path["actions"]
            ] # convert from one-hot to integers
            good_actions = list(best_path_actions[:np.amin(final_reward_times)+1])
            self.master_actions = np.asarray(list(self.master_actions) + good_actions)

            logger.log("Discovered new actions \n %s\n to total rewards %d!"%\
                (good_actions,np.sum(best_path["raw_rewards"])),
                color="green",
            )
            use_default_reset = False
        print(self.master_actions)
        return use_default_reset

    def reset(self,env):
        assert isinstance(env,AtariEnv)
        assert env.max_start_nullops == 0 # cannot deal with stochastic env
        env.ale.reset_game()
        prior_reward = 0
        for a in self.master_actions:
            o, r, done, env_info = env.step(a)
            prior_reward += r
        env._prior_reward = prior_reward
        # save the restored state for debugging
        if len(self.master_actions) > 0 and self.restored_state_folder is not None:
            timestamp = get_time_stamp()
            filename = os.path.join(self.restored_state_folder,timestamp+".jpg")
            img = env.ale.getScreenRGB()[:,:,::-1]
            cv2.imwrite(filename, img)
            logger.log("Saved restored state to %s"%(filename))

    def get_param_values(self):
        params = dict(
            master_actions=self.master_actions
        )
        return params

    def set_param_values(self,params):
        self.master_actions = params["master_actions"]
