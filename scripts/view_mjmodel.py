from mjpy import MjViewer, MjModel
import numpy as np
import h5py

viewer = MjViewer()
model = MjModel("vendor/mujoco_models/swimmer.xml")

data = h5py.File("/Users/dementrock/research/control/domain_data/mjc_examples/ex_swimmer.h5")
xs = data['xs'].value
us = data['us'].value
#import ipdb; ipdb.set_trace()

print "== Mujoco model info"
print "DOF:", model.data.qpos.size
print "#Actuators:", model.data.ctrl.size

viewer.set_model(model)
viewer.start()



import time
time.sleep(3)
#one_hot = np.zeros_like(model.data.qpos)
#one_hot[4,0] += 1
#one_hot[3,0] += 1
#model.data.qpos = one_hot
##print "state:", model.data.qpos
#model.forward()
#model.data.qpos 
model.data.qpos = xs[0][:5]
model.data.qvel = xs[0][5:]
model.forward()

for idx in range(len(xs)):#0):#1000):#1000):
    #for __ in range(30):
    #    model.step()
    #model.data.ctrl = us[idx]# / 25# / 10
    #import ipdb; ipdb.set_trace()
    model.data.qpos = xs[idx][:5]
    model.data.qvel = xs[idx][5:]
    model.forward()
    #for _ in range(25):
    #    model.step()
    viewer.loop_once()
    #time.sleep(0.1)
while True:
    viewer.loop_once()
#while not viewer.should_stop():
#    for _ in range(10):
#        model.step()
#    print model.data.qpos.reshape(-1)
#    viewer.loop_once()
