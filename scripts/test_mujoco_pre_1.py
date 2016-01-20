from rllab.mjcapi.rocky_mjc_pre_1.mjlib import mjlib
from rllab.mjcapi.rocky_mjc_pre_1.mjtypes import MjModelWrapper, MjDataWrapper
import mjcpy

model = MjModelWrapper(mjlib.mj_loadModel("src/mjc/mjcdata/3swimmer.bin"))

data = MjDataWrapper(mjlib.mj_makeData(model.ptr), model)

from rllab.mjcapi.rocky_mjc_1_22.mjlib import mjlib as mjlib2
from rllab.mjcapi.rocky_mjc_1_22.mjtypes import MjModelWrapper as MjModelWrapper2, MjDataWrapper as MjDataWrapper2

model2 = MjModelWrapper2(mjlib2.mj_loadXML("vendor/mujoco_models/1_22/swimmer.xml", ""))
data2 = MjDataWrapper2(mjlib2.mj_makeData(model2.ptr), model2)


#print "should be equal"
#print model.body_inertia
#print model2.body_inertia
#
#print "should be equal"
#print model.body_mass
#print model2.body_mass

world = mjcpy.MJCWorld("src/mjc/mjcdata/3swimmer.bin")
world_info = world.GetModel()

import numpy as np
init_state = np.zeros(10)
actuated_joints = [3,4]
actuated_dims = np.array([dofidx for (dofidx,jntidx) in enumerate(world_info["dof_jnt_id"]) if jntidx in actuated_joints])
world.SetActuatedDims(actuated_dims)
world.SetContactType(mjcpy.ContactType.values[mjcpy.ContactType.SPRING.real])

t = 100#1#5#10
state = np.zeros(10)
for _ in range(t):
    state = world.Step(state, np.ones(2)*50)#init_state, np.zeros(1))
print state

#data2.ctrl = np.ones(2)
for _ in range(t):
    data2.ctrl = np.ones(2)*50
    mjlib2.mj_step(model2.ptr, data2.ptr)
print np.concatenate([data2.qpos, data2.qvel]).flatten()
#print data2.qvel
#model2
#import ipdb; ipdb.set_trace()
# MjModel("vendor/mujoco_models/1_22/swimmer.xml")
