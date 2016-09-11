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
from rllab.misc import logger


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
        self.images = None
        self.probs = None
        self.ale_ids = None
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
        ale_ids = np.concatenate([path["env_infos"]["ale_ids"] for path in paths])

        self.candidates = np.asarray([
            internal_states[i] for i in range(len(is_terminals))
            if is_terminals[i] == False
        ])
        self.ale_ids = np.asarray([
            ale_ids[i] for i in range(len(is_terminals))
            if is_terminals[i] == False
        ])
        counts = np.asarray([
            counts[i] for i in range(len(is_terminals))
            if is_terminals[i] == False
        ])
        self.probs = self.counts_to_probs(counts)

        if "rgb_images" in paths[0]["env_infos"]:
            imgs = np.concatenate([path["env_infos"]["rgb_images"] for path in paths])
            self.images = np.asarray([
                imgs[i] for i in range(len(is_terminals))
                if is_terminals[i] == False
            ])


    def reset(self,env):
        """
        Called by the environment to reset to a state different from the default
        """
        assert isinstance(env,AtariEnv)
        if (self.candidates is None) or (np.random.uniform() > self.p):
            # initially it doesn't have any candidates
            env.ale.reset_game()
            logger.log("Reset to default initial state")
            use_default_reset = True
        else:
            # only reset to the candidates discovered by this env
            candidates = []
            probs = []
            legal_id = hex(id(env.ale))
            for state,prob,ale_id in zip(self.candidates, self.probs, self.ale_ids):
                if ale_id == legal_id:
                    candidates.append(state)
                    probs.append(prob)
            probs = np.asarray(probs) / np.sum(probs) # renormalize

            dist = np.random.multinomial(1,probs)
            index = np.argwhere(dist == 1)[0][0]
            state = int(candidates[index]) # int64 -> int
            env.ale.restoreState(state)
            # assert env.ale.cloneState() == state # doesn't work
            logger.log("Restored to state %s"%(state))

            # debug: print the original image and the image after restoration
            if self.restored_state_folder is not None:
                env.ale.act(0) # refresh the screen
                timestamp = get_time_stamp()
                filename = os.path.join(self.restored_state_folder,timestamp+".jpg")
                img = env.ale.getScreenRGB()[:,:,::-1]
                cv2.imwrite(filename, img)

                if self.images is not None:
                    images = []
                    for img,ale_id in zip(self.images,self.ale_ids):
                        if ale_id == legal_id:
                            images.append(img)

                    true_filename = os.path.join(self.restored_state_folder,timestamp+"_true.jpg")
                    true_img = images[index][:,:,::-1]
                    cv2.imwrite(true_filename,true_img)
            use_default_reset = False
        return use_default_reset

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
            images=self.images,
            ale_ids=self.ale_ids,
        )
        return params

    def set_param_values(self,params):
        self.candidates = params["candidates"]
        self.probs = params["probs"]
        self.p = params["p"]
        self.exponent = params["exponent"]
        self.images = params["images"]
        self.ale_ids = params["ale_ids"]
