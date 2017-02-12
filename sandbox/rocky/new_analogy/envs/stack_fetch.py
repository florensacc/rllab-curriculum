from gpr.envs.fetch.sim_fetch import Getter
from gpr.reward import *
from gpr.env import Env
from gpr.worldgen import WorldBuilder
from gpr.env import Env
from gpr.worldgen import WorldBuilder
from gpr.worldgen.objs.geom import Geom
from gpr.worldgen.objs.floor import Floor
from gpr.worldgen.objs.obj_from_xml import ObjFromXML
from gpr.worldgen.objs.obj import Obj
from gpr.trajectory import trajectory_good_10x
from gpr.core import BaseExperiment, OptimizerParams, WorldParams
import os


def dist(from_id, relation, to_id, metric):
    if relation == "top":
        return (DistReward("bottom_left%d" % from_id, "top_left%d" % to_id, metric) + \
                DistReward("bottom_right%d" % from_id, "top_right%d" % to_id, metric)) * 0.5
    elif relation == "right":
        return (DistReward("top_right%d" % from_id, "top_left%d" % to_id, metric) + \
                DistReward("bottom_right%d" % from_id, "bottom_left%d" % to_id, metric)) * 0.5
    else:
        assert (False)


class Experiment(BaseExperiment):
    def __init__(self, nboxes=5, horizon=50, num_substeps=1, mocap=True, obs_type="flatten"):
        self.nboxes = nboxes
        self.horizon = horizon
        self.num_substeps = num_substeps
        self.mocap = mocap
        self.obs_type = obs_type

    def make(self, task_id=None):
        world_params = WorldParams(obs_type=self.obs_type,
                                   # bound=((-0.5, 0.5), (-0.5, 0.5), (0.0, 0.6)),
                                   bound=((-0.75, 0.75), (-0.75, 0.75), (0.0, 1.6)),# \
                                   randomize_material=True,
                                   randomize_velocity=0.,
                                   randomize_robot_position=0.0,
                                   randomize_objects_position=1.0,
                                   num_substeps=self.num_substeps,
                                   marker_size=0.01)
        builder = WorldBuilder(world_params)
        floor = Floor()
        robot = ObjFromXML("robot/fetch", mocap=self.mocap, markers2keep="grip")
        # robot = ObjFromXML("robot/copter-claw")
        builder.append(floor)
        # floor.append(robot)
        floor.append(robot, (-0.30, 0.0, 0))
        robot.get_object_size = Getter((0.6, 0.528, 1.093))  # Hack to place objects in front of the robot.

        top = Geom("box", (0.405, 0.6, 0.4), mass=2000, no_joints=True)
        floor.append(top, (0.25, 0, 0))

        smoothness_reward = 1e3 * -PenaltyReward('active_contacts_efc_pos') + 1e-5 * -PenaltyReward('qfrc_actuator')
        reward_sequence = SequenceReward()
        geoms = []
        for i in range(self.nboxes):
            geom = Geom("box", 0.05)
            geoms.append(geom)
            geom.mark("geom%d" % i, "inside")
            geom.mark("top_left%d" % i, "top_left")
            geom.mark("top_right%d" % i, "top_right")
            geom.mark("bottom_left%d" % i, "bottom_left")
            geom.mark("bottom_right%d" % i, "bottom_right")
            top.append(geom)

        if task_id is None:
            task_id = self.sample_task_id()

        metric = "GPS:1,0.3,1e-4"
        assert (max([max(p[0], p[2]) for p in task_id]) < self.nboxes)
        assert (min([min(p[0], p[2]) for p in task_id]) >= 0)
        for i in range(len(task_id)):
            r = -DistReward("grip", "geom%d" % task_id[i][0], metric) + \
                smoothness_reward + (i - 1) * 100.0

            for from_id, relation, to_id in task_id[:(i + 1)]:
                r -= dist(from_id, relation, to_id, metric)

            # Makes sure that we are not moving any other box.
            # for g in range(self.nboxes):
            #    if g != task_id[i][0]:
            #        for k in range(3):
            #            r -= PenaltyReward('joint_%s:slide%d_qvel' % (geoms[g].name, k), metric)
            reward_sequence.append(r, LessCond(dist(task_id[i][0], task_id[i][1], task_id[i][2], "L1"), 0.015))
        return Env(world_builder=builder, reward=reward_sequence, horizon=self.horizon, task_id=task_id)

    def sample_task_id(self):
        ret = []
        ret.append([(1, "right", 0), (2, "top", 1), (3, "top", 0), (4, "top", 3)])
        ret.append([(1, "right", 0), (2, "top", 1), (3, "top", 1), (4, "top", 3)])
        ret.append([(1, "right", 0), (2, "right", 1), (3, "right", 2), (4, "right", 3)])
        ret.append([(1, "top", 0), (2, "top", 1), (3, "top", 2), (4, "top", 3)])
        return ret[self.task_id_seed.randint(0, len(ret))]

    @property
    def optimizer_params(self):
        return OptimizerParams(pre_lqr=0, post_lqr=0)


if __name__ == "__main__":
    from gpr.runner import SimpleRunner
    expr = Experiment(nboxes=2, horizon=1000, mocap=False, num_substeps=1)
    env = expr.make(task_id=[[1, "top", 0]])
    env.reset()
    while True:
        env.render()
        import time
        time.sleep(0.01)
    # SimpleRunner(env).run()
