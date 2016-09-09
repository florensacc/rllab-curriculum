"""
Notice:
1. we are not implementing importance sampling to correct for the difference between true initial state and prioritized states. This was done in prioritized replay, but not very implementable here.
2. We may try keeping some past undervisited states in case they will be useful forthe current iteration.
3. While updating, the states from different paths may be the same. Just don't care now.
"""
import numpy as np
import os
import cv2
from sandbox.haoran.myscripts.myutilities import get_time_stamp
from sandbox.haoran.hashing.bonus_trpo.envs.atari_env import AtariEnv


class AtariCountResetter(object):
    def __init__(self,
            p=0.5,
            exponent=1,
            restored_state_folder="/tmp/restored_state"
        ):
        """
        p: probability that the environment is set to a different state from the default
        """
        self.p = p
        self.exponent = exponent
        self.candidates = None
        self.probs = None
        self.candidate_env_ids = None
        self.restored_state_folder = restored_state_folder
        if restored_state_folder is not None:
            if not os.path.isdir(restored_state_folder):
                os.system("mkdir -p %s"%(restored_state_folder))

    def update(self,paths):
        """
        Rearrange the priorities of different states based on their visit counts
        """
        internal_states = np.concatenate([path["env_infos"]["internal_states"] for path in paths])
        is_terminals = np.concatenate([path["env_infos"]["is_terminals"] for path in paths])
        counts = np.concatenate([path["counts"] for path in paths])

        self.candidates = [
            internal_states[i] for i in range(len(is_terminals))
            if is_terminals[i] == False
        ]
        counts = np.asarray([
            counts[i] for i in range(len(is_terminals))
            if is_terminals[i] == False
        ])
        self.probs = self.counts_to_probs(counts)


    def reset(self,env):
        """
        Called by the environment to reset to a state different from the default
        """
        assert isinstance(env,AtariEnv)
        if (self.candidates is None) or (np.random.uniform() > self.p):
            # initially it doesn't have any candidates
            env.ale.reset_game()
            print("Reset to default initial state")
        else:
            # only reset to the candidates discovered by this env
            # candidates = []
            # probs = []
            # for state,probin zip(self.candidates, self.probs, self.candidate_env_ids):
            #     # print("%s + %s"%(env_id,hex(id(env.ale))))
            #     # if env_id  == hex(id(env.ale)):
            #     candidates.append(state)
            #     probs.append(prob)
            # probs = np.asarray(probs) / np.sum(probs) # renormalize

            dist = np.random.multinomial(1,self.probs)
            index = np.argwhere(dist == 1)[0][0]
            state = int(self.candidates[index]) # int64 -> int
            env.ale.restoreState(state)
            env.ale.act(0)
            # if accidentally goes to a terminal state, use the default reset
            if env.ale.game_over():
                env.ale.reset_game()
            # assert env.ale.cloneState() == state # debug
            print("Restored to state %s"%(state))

            if self.restored_state_folder is not None:
                filename = get_time_stamp()
                filename = os.path.join(self.restored_state_folder,filename+".jpg")
                img = env.ale.getScreenRGB()[:,:,::-1]
                cv2.imwrite(filename, img)


    def counts_to_probs(self,counts):
        unnormalized_probs = counts ** (-self.exponent)
        probs = unnormalized_probs / np.sum(unnormalized_probs)
        return probs

    def get_param_values(self):
        params = dict(
            candidates=self.candidates,
            probs=self.probs,
            p=self.p,
            exponent=self.exponent,
        )
        return params

    def set_param_values(self,params):
        self.candidates = params["candidates"]
        self.probs = params["probs"]
        self.p = params["p"]
        self.exponent = params["exponent"]
