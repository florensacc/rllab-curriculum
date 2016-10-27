import pickle as p
import numpy as np


def load_attractor_data():
    base_path = '/Users/TheMaster/Desktop/Current_Work/exp/grab_copter-3/'
    one_traj = p.load(open(base_path + 'trajectory_params:_trial:0000995.pkl', 'rb'))
    #scenario = one_traj.scenario
    #start_pt = one_traj.observations_over_time()
    prev_step = np.expand_dims(one_traj.solution['qpos'][1], 0)
    cur_step = np.expand_dims(one_traj.solution['qpos'][2], 0)
    desired_step = np.expand_dims(one_traj.solution['qpos'][3], 0)
    #seconds_step = np.expand_dims(one_traj.solution['qpos'][1], 0)
    dt = 0.005

    wts = one_traj.scenario.world.model.actuator_invweight0
    qvel_desired = np.expand_dims(one_traj.solution['qvel'][2], 0)
    q_acc_desired = (prev_step[0, :] - 2*cur_step[0, :] + desired_step[0, :])/(dt*dt)
    q_acc_desired = np.expand_dims(q_acc_desired, 0)

    force_actuated_dofs, force_unactuated_dofs, state_dict = one_traj.scenario.world.model.invstep(cur_step,
                                                                                                   qvel_desired,
                                                                                                   q_acc_desired)

    force_actuated_dofs = force_actuated_dofs/wts
    true_f_act = one_traj.solution['ctrl'][2]

    klklk
    #print(one_traj.actions_over_time())
    #print(one_traj.solution['cost'])
    #print('done')

if __name__ == "__main__":
    load_attractor_data()