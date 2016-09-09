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
        counts = np.concatenate([path["counts"] for path in paths])
        env_ids = np.concatenate([path["env_infos"]["env_ids"] for path in paths])

        self.candidates = internal_states
        self.probs = self.counts_to_probs(counts)
        self.candidate_env_ids = env_ids


    def reset(self,env):
        """
        Called by the environment to reset to a state different from the default
        """
        # assert(isinstance(env,AtariEnv))
        if (self.candidates is None) or (np.random.uniform() > self.p):
            # initially it doesn't have any candidates
            env.ale.reset_game()
        else:
            # only reset to the candidates discovered by this env
            candidates = []
            probs = []
            for state,prob,env_id in zip(self.candidates, self.probs, self.candidate_env_ids):
                # print("%s + %s"%(env_id,hex(id(env.ale))))
                # if env_id  == hex(id(env.ale)):
                candidates.append(state)
                probs.append(prob)
            probs = np.asarray(probs) / np.sum(probs) # renormalize

            dist = np.random.multinomial(1,probs)
            index = np.argwhere(dist == 1)[0][0]
            state = candidates[index]
            env.ale.restoreState(state)

            print "Restoring to state %s"%(state.obj)
            if self.restored_state_folder is not None:
                filename = get_time_stamp()
                filename = os.path.join(self.restored_state_folder,filename+".jpg")
                img = env._get_image()
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
            candidate_env_ids=self.candidate_env_ids
        )
        return params

    def set_param_values(self,params):
        self.candidates = params["candidates"]
        self.probs = params["probs"]
        self.p = params["p"]
        self.candidate_env_ids = params["candidate_env_ids"]
        self.exponent = params["exponent"]
