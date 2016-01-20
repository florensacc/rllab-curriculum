from rllab.mjcapi.rocky_mjc_pre_1.mjlib import mjlib
from rllab.mjcapi.rocky_mjc_pre_1.mjtypes import MjModelWrapper, MjDataWrapper

model = MjModelWrapper(mjlib.mj_loadModel("src/mjc/mjcdata/3swimmer.bin"))
data = MjDataWrapper(mjlib.mj_makeData(model.ptr), model)

from rllab.mjcapi.rocky_mjc_1_22.mjlib import mjlib as mjlib2
from rllab.mjcapi.rocky_mjc_1_22.mjtypes import MjModelWrapper as Wrapper2

model2 = Wrapper2(mjlib2.mj_loadXML("vendor/mujoco_models/1_22/swimmer.xml", ""))

import ipdb; ipdb.set_trace()

# MjModel("vendor/mujoco_models/1_22/swimmer.xml")
