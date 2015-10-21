from mjpy import MjViewer, MjModel
import numpy as np

viewer = MjViewer()
model = MjModel("vendor/mujoco_models/locomotion.xml")

print "== Mujoco model info"
print "DOF:", model.data.qpos.size
print "#Actuators:", model.data.ctrl.size

viewer.set_model(model)
viewer.start()



one_hot = np.zeros_like(model.data.qpos)
#one_hot[1,0] += 1
model.data.qpos = one_hot
#print "state:", model.data.qpos
model.forward()
for _ in range(0):#1000):#1000):
    for __ in range(30):
        model.step()
    viewer.loop_once()
    #time.sleep(0.1)
while True:
    viewer.loop_once()
#while not viewer.should_stop():
#    for _ in range(10):
#        model.step()
#    print model.data.qpos.reshape(-1)
#    viewer.loop_once()
