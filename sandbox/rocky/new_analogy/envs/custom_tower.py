from gpr.reward import *
from gpr.env import Env
from gpr.worldgen import WorldBuilder
from gpr.env import Env
from gpr.worldgen import WorldBuilder
from gpr.worldgen.objs.geom import Geom
from gpr.worldgen.objs.material import Material
from gpr.worldgen.objs.floor import Floor
from gpr.worldgen.objs.skybox import Skybox
from gpr.worldgen.objs.obj_from_xml import ObjFromXML
from gpr.worldgen.objs.obj import Obj
from gpr.trajectory import trajectory_good_10x
from gpr.core import BaseExperiment, OptimizerParams, WorldParams
import itertools


def dist(from_id, to_id, metric):
    return DistReward("geom%d" % from_id, "geom%d" % to_id, metric)


class Experiment(BaseExperiment):

    def __init__(self, nboxes=5, horizon=50, bound=((-0.75, 0.75), (-0.75, 0.75), (0.0, 1.6))):
        self.nboxes = nboxes
        self.horizon = horizon
        self.bound = bound

    def make(self, task_id=None):
        world_params = WorldParams(obs_type='full_state', \
                                   randomize_material=True, \
                                   solver_iterations=200, \
                                   timestep=0.002, \
                                   solver="CG", \
                                   bound=self.bound, \
                                   randomize_velocity=0., \
                                   randomize_robot_position=0.0, \
                                   randomize_objects_position=1.0, \
                                   num_substeps=5,
                                   marker_size=0.01)
        floor = Floor(Material(texture="textures/wood.png", texture_type="2d"))
        robot = ObjFromXML("robot/copter-claw")
        floor.append(robot)
        builder = WorldBuilder(world_params)
        builder.append(Skybox())
        builder.append(floor)

        geoms = []
        for i in range(self.nboxes):
            char = chr(ord('A') + i % 26)
            geom = Geom("box", 0.05, material=Material(texture="textures/chars/" + char + ".png"))
            geoms.append(geom)
            geom.mark("geom%d" % i, "inside")
            floor.append(geom)
        reward_sequence = SequenceReward()
        if task_id is None:
            task_id = self.sample_task_id()

        metric = "GPS:1,0.3,1e-4"
        smoothness_reward = 1e3 * -PenaltyReward('active_contacts_efc_pos') + 1e-6 * -PenaltyReward('qfrc_actuator')

        count = 0
        for sub_task in task_id:
            for i in range(1, len(sub_task)):
                r = -DistReward("grip", "geom%d" % sub_task[i], metric) + smoothness_reward + count * 30.0
                for j in range(1, i + 1):
                    r -= dist(sub_task[j], sub_task[j - 1], metric)
                l2 = dist(sub_task[i], sub_task[i - 1], "L2")
                reward_sequence.append(r, LessCond(l2, 0.06 * 0.06))
                count += 1
        return Env(world_builder=builder, \
                   reward=reward_sequence, \
                   horizon=self.horizon, \
                   task_id=task_id)

    def sample_task_id(self):
        count = self.task_id_seed.randint(2, 5)
        ret = []
        for _ in range(count):
            height = self.task_id_seed.randint(2, 5)
            ret.append(self.task_id_seed.permutation(self.nboxes)[:height])
        return ret

    @property
    def optimizer_params(self):
        return OptimizerParams(pre_lqr=0, \
                               post_lqr=0, \
                               save_intermediate=True, \
                               particles=100, \
                               mpc_horizon=10, \
                               mpc_steps=5, \
                               skip=1)