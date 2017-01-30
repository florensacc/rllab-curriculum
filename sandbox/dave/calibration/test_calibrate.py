import numpy as np
from sandbox.dave.calibration.calibration_utils import *
from sandbox.dave.calibration.xml_utils import *

def test_calibrate():
    groundtruth_params = {'shoulder_pan_joint': {'actuator_kp': 0.3}}    
    env = build_env(groundtruth_params)
    obs = env.reset()[0]
    control_sequence = [obs]
    shoulder_idx = calibration_config.joint_indices['shoulder_pan_joint']
    ctrl_magnitudes = [0.05, 0.1, 0.15]
    for ctrl_magnitude in ctrl_magnitudes:
        for i in range(100):
            new_control = control_sequence[-1].copy()
            if (i // 10) % 2 == 0:
                new_control[shoulder_idx] += ctrl_magnitude
            else:
                new_control[shoulder_idx] -= ctrl_magnitude
            control_sequence.append(new_control)
   
    control_sequence = np.expand_dims(np.stack(control_sequence), 0)
    control_sequence = control_sequence[:, :, 3:12]

    qpos = rollouts(env, control_sequence)
    train_data = {'shoulder_pan_joint': {'qpos': qpos, 'ctrl': control_sequence}}
    
    params = {'shoulder_pan_joint': ['actuator_kp']}
    optimized_params = calibrate_xml(train_data, params)
    
    baseline = evaluate_xml(train_data)['shoulder_pan_joint']
    result = evaluate_xml(train_data, optimized_params)['shoulder_pan_joint']
    tolerance = 1e-5
    assert(result < tolerance)
    assert(np.abs(optimized_params['shoulder_pan_joint']['actuator_kp'] - \
                  groundtruth_params['shoulder_pan_joint']['actuator_kp']) < tolerance)

