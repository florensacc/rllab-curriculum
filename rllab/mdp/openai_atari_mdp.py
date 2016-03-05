from rl.envs.atari import AtariEnv, OBS_RAM, OBS_IMAGE
from rllab.mdp.base import MDP
from rllab.misc import autoargs
from rllab.core.serializable import Serializable
import theano

# pylint: disable=no-member
floatX = theano.config.floatX
# pylint: enable=no-member


class AtariMDP(MDP, Serializable):

    @autoargs.arg("rom_name", type=str,
                  help="Name of the game ROM, without the file type extension")
    @autoargs.arg("obs_type", type=str, choices=["ram", "image"],
                  help="Observation type. Must be either ram or image.")
    @autoargs.arg("frame_skip", type=int,
                  help="How many frames to skip between each step.")
    def __init__(self, rom_name="pong", obs_type="ram", frame_skip=4):
        Serializable.quick_init(self, locals())
        int_obs_type = OBS_RAM if obs_type == "ram" else OBS_IMAGE
        self._mdp = AtariEnv(
            rom_name=rom_name,
            obs_type=int_obs_type,
            frame_skip=frame_skip
        )

    def reset(self):
        raw_obs = self._mdp.reset().flatten()
        return self._normalize_obs(raw_obs)

    def step(self, action):
        ret = self._mdp.step(action)
        return self._normalize_obs(ret.observation.flatten()), \
            ret.reward, ret.done

    def _normalize_obs(self, obs):
        return (obs / 255.0) * 2 - 1.0

    @property
    def action_dim(self):
        return self._mdp.action_space.n

    @property
    def action_dtype(self):
        return 'uint8'

    @property
    def observation_dtype(self):
        return floatX

    @property
    def observation_shape(self):
        return self._mdp.observation_space.shape

    def plot(self):
        self._mdp.render()
