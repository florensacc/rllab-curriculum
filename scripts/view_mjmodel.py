from mjpy import MjViewer, MjModel
import numpy as np

viewer = MjViewer()
model = MjModel("vendor/mujoco_models/gripper.xml")

viewer.set_model(model)
viewer.start()

#model.data.qpos = np.array([[1.57, 1.57]])
model.forward()
for _ in range(10):
    for __ in range(30):
        model.step()
    viewer.loop_once()
while True:
    viewer.loop_once()
#while not viewer.should_stop():
#    for _ in range(10):
#        model.step()
#    print model.data.qpos.reshape(-1)
#    viewer.loop_once()
