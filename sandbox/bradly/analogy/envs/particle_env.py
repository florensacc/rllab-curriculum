from __future__ import print_function
from __future__ import absolute_import
from conopt.env import Env
from conopt.worldgen import WorldBuilder
from conopt.worldgen.objs import ObjFromXML, Floor
from conopt import cost


class ParticleEnv(Env):
    def __init__(self, seed=None):
        global_properties = {
            "bound": ((-0.5, 0.5), (-0.5, 0.5), (0.0, 2.5)),
            "randomize_velocity": True
        }
        builder = WorldBuilder(seed, global_properties)
        floor = Floor()
        builder.append(floor)
        obj = ObjFromXML("robot/particle")
        floor.append(obj)
        obj.mark("object", "inside")
        floor.mark("target", "top", True, False, [0, 0, 1])
        world = builder.to_world(obs_type="full_state")
        c = cost.DistCost("object", "target") + \
            3e-2 * cost.PenaltyCost("ctrl")
        self.agent_name = "object"
        self.target_name = "target"
        Env.__init__(self, world=world, reward_fn=-c, batchsize=1)
