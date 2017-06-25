""" Hyperparameters for MJC peg-in-hole with PR2 model. """
from __future__ import division

import os.path
from datetime import datetime

from algorithm.cost.cost_sum import CostSum
from algorithm.mpc_opt.cost_offset_perturbation import *
from gps import __file__ as gps_filepath
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_mpc import CostMPC
from gps.algorithm.cost.cost_torque import CostTorque
from gps.algorithm.cost.cost_utils import RAMP_CONSTANT
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.mpc_opt.cost_offset import *
from gps.algorithm.policy.gaussian_noise import *
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.gui.config import generate_experiment_info
from gps.oc_refactor.fd_dynamics import FDDynamics
from gps.oc_refactor.online_dynamics import NoPrior
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, ACTION, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES


SENSOR_DIMS = {
    JOINT_ANGLES: 7,
    JOINT_VELOCITIES: 7,
    END_EFFECTOR_POINTS: 9,
    END_EFFECTOR_POINT_VELOCITIES: 9,
    ACTION: 7,
}
# theta = -np.pi/2
theta = -np.pi/2
d = 0.15
PR2_GAINS = np.array([3.09, 1.08, 0.393, 0.674, 0.111, 0.152, 0.098])
# EE_TGT = np.array([0.0, 0.3, -0.5,  0.0, 0.3, -0.2, 0.0, 0.45, -0.35])
EE_TGT = np.array([0.0, 0.3, -0.45-d,  0.0, 0.3, -0.15-d, 0.0+0.15*np.sin(theta), 0.3+0.15*np.cos(theta), -0.3-d])
WP = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0])

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/mjc_key/'

CONDITIONS = 1

common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': CONDITIONS,
    'filename': ['./mjc_models/pr2_arm3d_key.xml'],
    # convention : first indices are training, last are testing
    'train_conditions': [i for i in range(CONDITIONS)],
    'test_conditions': [],
    'iterations': 10,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentMuJoCo,
    'filename': common['filename'],
    'x0': [np.concatenate([np.array([0.1, 0.1, -1.54, -1.7, 1.54, -0.2, 0]),
                          np.zeros(7)])],
    'dt': 0.05,
    'substeps': 5,
    'conditions': common['conditions'],
    'T': 100,
    'pos_body_idx': np.array([1]),
    'pos_body_offset': np.array([0, 0.0, 0]),
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [],
    'meta_include': [],
    'camera_pos': np.array([1., 2., 1.5, 0., 0., 0.]),
}

mpc = {
    'dX': 32,                       # 7+7+6+6,
    'dU': 7,
    'dT': 39,
    'tgt': np.zeros(32),
    'warm_start': False,
    'u_clip': 3.0,
    'H': 10,
    'hindsight_H':  100,
    'hindsight_H_delta': 0,
    'lqr_discount': 0.99,
    'init_test_runs': 0,            # number of runs for initializing prior
    'hindsight_max_samples': 200,    # maximum samples (trajectories) to save
    'hindsight_min_samples': 1,     # minimum samples (trajectories) to start adding cost offset
    'no_offset_runs': 0,
    'success_threshold': 1e6, #0.07,
    'failure_threshold': 1e6,
    'nn_multiproc': False,
    #########################################
    'noise_generator': {
        'type': ImprovementThresholdedGaussianNoise,
        'dU': 7,
        'scale': 0.005,         # small random noise on actions (just for stochastic trajectories)
        # 'scale': 0.002,         # small random noise on actions (just for stochastic trajectories)
        'burst_scale': 0.0,     # no burst noise on actions - we are perturbing the cost offset instead
        'burst_len': 25,         # short burst length
        'burst_relax': 10,      # relax period after burst
        'window_len': 30,       # window to calculate reward not improving (to initiate burst)
        'max_bursts': 1,        # only 1 burst per episode
        'noise_decay': 0.99,
        'smooth_noise': True,
        'smooth_noise_var': 5.0,
        'smooth_noise_renormalize': True,
        'success_threshold': 0.01,
        'eetgt': EE_TGT,
        'wp': WP,
        'ee_idx': slice(14, 23),
        'ee_vel_idx': slice(23, 32)
    },
    #########################################
    'cost_offset': {
        'max_steps': 200,               # turn off offset after max_steps time steps
        'loss_wu': PR2_GAINS,           # weight of actions in the action similarity loss function
        'input_x_ind': slice(14, 23),   # EE points
        'offset_NN': CostOffsetNN,
        'hidden_sizes': [10, 10],
        'activation': [TT.tanh, TT.tanh, None],
        'function_list_op': 'replace',  # don't change this (can be used to learn multiple offset)
        'offset_error': 'single',       # don't change this (can be used to learn multiple offset)
        'optimization_options': {
            'maxiter': 100,
            'disp': True
        },
        'perturbation_len': 25,
        'sample_size_factor': 0.5,
    },
    # NN that generates a *random* offset perturbation used during exploration
    'cost_offset_noise_generator': {
        'type': CostOffsetPerturbation,
        'offset_NN': CostOffsetConstant,
        'variance': 0.25,
    },
    # NN that learns a mapping from state to perturbation (by RWR directly on the perturbation)
    'cost_offset_learned_perturbation': {
        'type': CostOffsetPerturbation,
        'offset_NN': CostOffsetNN,
        'contextual': False  # if true, learning is done only from the perturbation initial state. Otherwise,
        # the full trajectory is used (even though the perturbation is the same throughout it)
    },
    'training_type': 'behavioral_cloning',  # behavioral_cloning / direct training (see above definitions)
    # 'training_type': 'direct',
    #######################################
    'lambda_params': 0.0001,
    'lambda_trust': 0.0,
    'final_penalty': 1.,
    'rl_weight_coeff': 5.0,
    'save_data_at_iterations': [0],
    'visualize_vf': False,
    'l1': 0.0,
    'l2': 1.0,
    'alpha': 1e-5,
    'jnt_idx': slice(0, 7),
    'ee_idx': slice(14, 23),
    'eetgt': EE_TGT,   # End-effector target
    'wu': 2e-3/PR2_GAINS,  # Torque penalty
    'wp': WP,
    'wpm_mult': 1.,
    'use_jacobian': False,
    'cost': CostMPC,
    'ramp_option': RAMP_CONSTANT,  # ramp within T
    #########################################
    'dynamics': {
        'type': FDDynamics,
        'agent': agent,
        'dX': 32,
        'dU': 7,
        'fd_epsilon': 1e-4,             # finite dynamics epsilon
        'sigreg': 1e-6,
        'dyn_init_from_data': True,     # initialize online dynamics from prior data
        'dyn_init_mu': np.zeros(39+32),
        'dyn_init_sig': np.eye(39+32),
        'adaptive': False,              # fixed gamma
        'init_gamma': .1,               # Higher means update faster. Lower means keep more history.
        'min_gamma': .1,                # not used with fixed gamma
        'eta_0': 20.,                   # not used with fixed gamma
        'init_N': 0.,                   # not used with fixed gamma
        'max_N': 16.,                   # not used with fixed gamma
        'min_N': .1,                    # not used with fixed gamma
        'nu_0': 1.,                     # not used with fixed gamma
        'future_prediction_steps': 1,   # hindsight dynamics prediction - predict dynamics at t+future_prediction_steps
    },
    #########################################
    'init_prior_class': NoPrior,
    'prior': {
        'init_prior_data': common['data_files_dir']+'offline_dynamics_data.mat',
        'type': DynamicsPriorGMM,
        'max_clusters': 10,
        'min_samples_per_cluster': 40,
        'max_samples': 40,
        'strength': 1.0,
        'gmm': {
            'warmstart': False,
        },
    },
    'data_files_dir': common['data_files_dir'],
}

#############################################################
# The following parameters are for the offline iLQG algorithm
#############################################################

algorithm = {
    'type': AlgorithmTrajOpt,
    'conditions': common['conditions'],
    'train_conditions': common['train_conditions'],
    'test_conditions': common['test_conditions'],
    'iterations': common['iterations'],
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_var': 1.0,
    'stiffness': 1.0,
    'stiffness_vel': 0.5,
    'dt': agent['dt'],
    'T': agent['T'],
}

torque_cost = {
    'type': CostTorque,
    'wu': 5e-5/PR2_GAINS,
}

fk_cost = {
    'type': CostFK,
    'end_effector_target': EE_TGT,
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-5,
    'wp': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [torque_cost, fk_cost],
    'weights': [1.0, 1.0],
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 40,
        'max_samples': 40,
        'strength': 1,
        'gmm': {
            'warmstart': False,
        },
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

algorithm['policy_opt'] = {}


config = {
    'iterations': algorithm['iterations'],
    'num_samples': 5,
    'verbose_trials': 1,
    'verbose_policy_trials': 0,
    'common': common,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
}

common['plot_controller_dist'] = False
common['plot_dynamics_prior'] = True
common['plot_clusters'] = True
common['info'] = generate_experiment_info(config)

