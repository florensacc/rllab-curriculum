from __future__ import print_function
from __future__ import absolute_import
from conopt.env import Env
from conopt.worldgen import WorldBuilder
from conopt.worldgen.objs import ObjFromXML, Floor
from conopt import cost
import numpy as np


class BiparticleEnv(Env):
    def __init__(self, seed=None, target_id=None):
        global_properties = {
            "bound": ((-1, 1), (-1, 1), (0.0, 2.5)),
            "randomize_velocity": True
        }
        builder = WorldBuilder(seed, global_properties)
        floor = Floor()
        builder.append(floor)
        obj = ObjFromXML("robot/particle")
        floor.append(obj)
        obj.mark("agent", "inside")
        if target_id is None:
            target_id = np.random.randint(low=0, high=2)
        floor.mark("target1", "top", True, False, [0, 0, 1])
        floor.mark("target2", "top", True, False, [0, 1, 0])
        target = ["target1", "target2"][target_id]
        world = builder.to_world(obs_type="full_state")
        c = cost.DistCost("agent", target) + \
            3e-2 * cost.PenaltyCost("ctrl")
        self.agent_name = "agent"
        self.target_name = target
        self.target_id = target_id
        Env.__init__(self, world=world, reward_fn=-c, batchsize=1)
