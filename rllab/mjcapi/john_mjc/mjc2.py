from rllab.mjcapi.john_mjc.config import floatX,CTRL_ROOT
from rllab.mjcapi.john_mjc.base_mdp import MDP
from rllab.mjcapi.john_mjc.mjc2_info import *
import mjcpy2
import numpy as np, os.path as osp
import random

def humanoid_initializer(x, properties):
    # initialize humanoid such that the knees are a bit bent
    alpha = properties["knee_bending"]
    x[12] = alpha
    x[13] = 2*alpha
    x[15] = alpha

    x[18] = alpha
    x[19] = 2*alpha
    x[21] = alpha
    # make one knee stick forward
    beta = properties["forward_init_parameter"]
    if random.random() <= 0.5:
        x[12] += beta
    else:
        x[18] += beta

def humanoid_initializer_tracking(x, properties):
    x[0] = np.random.randn(1)[0]
    x[1] = np.random.randn(1)[0]


def standing_initializer(x, properties):
    # x[12] = -2
    # x[18] = -2
    # x[13] = -2.2
    # x[19] = -2.2
    x[:] = np.array([ -0.315,   0.899,   0.085,   0.046,  -0.667,  -0.01,   -0.744,  -0.596,   0.302,
   -0.516,  -0.022,   0.019,   0.784,   0.332,  -0.307,  -0.332,  -0.099,  -0.023,
    1.072,   0.481,   0.302,   0.156,   0.804,   0.199,  -1.299,   0.63,    0.566,
    0.804,   0.061,   0.353,  -0.087,   0.38,   -0.54,    0.587,   0.945,   1.344,
   -0.781,  -5.014,  -0.629,   1.315,  -0.692,   0.771,   0.771,  -2.11,   -4.924,
    1.659,   0.055,   6.094,  -2.429,  -0.333, -10.076,   1.206,  -0.387,   0.7,
   -3.87 ])

def sitting_initializer(x, properties):
    """ x[2] = 1.05
    x[12] = -1.5
    x[18] = -1.5
    x[13] = -1.5
    x[19] = -1.5"""
    x[:] = np.array([ -0.5+-0.315,   -0.5 + 0.899,   0.085,   0.046,  -0.667,  -0.01,   -0.744,  -0.596,   0.302,
       -0.516,  -0.022,   0.019,   0.784,   0.332,  -0.307,  -0.332,  -0.099,  -0.023,
        1.072,   0.481,   0.302,   0.156,   0.804,   0.199,  -1.299,   0.63,    0.566,
        0.804,   0.061,   0.353,  -0.087,   0.38,   -0.54,    0.587,   0.945,   1.344,
       -0.781,  -5.014,  -0.629,   1.315,  -0.692,   0.771,   0.771,  -2.11,   -4.924,
        1.659,   0.055,   6.094,  -2.429,  -0.333, -10.076,   1.206,  -0.387,   0.7,
       -3.87 ])

def ant_initializer(x):
    x[0] = 4.0
    x[2] = 0.55
    x[8] = 1.0
    x[10] = -1.0
    x[12] = -1.0
    x[14] = 1.0



METADATA = {
    "default" : {
        "frame_skip": 1,
        "sample_halfrange" : 0.1,    
        "alive_bonus_coeff" : 1.0,
        "clip_features" : 1.0,
        "load_from_dot_in_file": False,
        "features" : ["cinert","cvel","qfrc_actuation","cfrc_ext","contactdists", "qfrc_constraint"]
    },
    "planar_humanoid" : {
        "world_file" : "planar_humanoid.xml",
        "has_ground" : True,
        "cost_type" : "locomotion_with_impact",
        "quad_ctrl_cost_coeff" : 1e-4,
        "lin_vel_cost_coeff" : 1.0,        
        "quad_impact_cost_coeff" : 0.0,
        "clip_impact_cost" : 10.0,        
        "obs_type" : "state+feats+dcom",
        "done_type" : "state_ranges",
        "allowed_state_ranges" : {
            0 : (1.0, 2.0),
            2 : (-1.0, 1.0),
        },
        "height_idx" : 0,
        "frame_skip" : 4,
    },
    "tars" : {
        "load_from_dot_in_file": False,
        "world_file" : "tars.xml", 
        "has_ground" : True,
        "experiment" : "walking", # can be "walking" or "standing"
        "cost_type" : "locomotion_with_impact_tars",
        "quad_ctrl_cost_coeff" : 1e-5,
        "lin_vel_cost_coeff" : 1.0,
        "upright_cost_coeff" : 1.0,
        "vel_deviation_cost_coeff" : 100.0,
        "quad_impact_cost_coeff" : 1e-5,
        "restingpos_cost_coeff" : 1e-3,
        "clip_impact_cost" : 10.0,        
        "obs_type" : "state+feats+dcom",
        "done_type" : "state_ranges",
        "quad_ctrl_cost": None,
        "allowed_state_ranges" : {
            # 2 : (0.74, 2.0),
            3: (0.7, 1.1)
        },
        "height_idx" : 2,
        "frame_skip": 1,
        "initializer": None
    },
    "ant" : {
        "load_from_dot_in_file": False,
        "world_file" : "ant.xml", 
        "has_ground" : True,
        "experiment" : "walking", # can be "walking" or "standing"
        "cost_type" : "locomotion_with_impact",
        "quad_ctrl_cost_coeff" : 1e-5,
        "find_ball_cost_coeff" : 1.0,
        "lin_vel_cost_coeff" : 1.0,
        "vel_deviation_cost_coeff" : 0.0,
        "quad_impact_cost_coeff" : 1e-5,
        "restingpos_cost_coeff" : 0.0,
        "clip_impact_cost" : 10.0,        
        "obs_type" : "state+feats+dcom",
        "done_type" : "state_ranges_ant",
        "quad_ctrl_cost": None,
        "allowed_state_ranges" : {
            2 : (0.2, 1.0),
        },
        "height_idx" : 2,
        "frame_skip": 1,
        # "initializer": None
        "initializer": lambda x, bending: ant_initializer(x),
    },
    "atlas" : {
        "load_from_dot_in_file": True,
        "world_file" : "atlas.xml", 
        "has_ground" : True,
        "experiment" : "walking", # can be "walking" or "standing"
        "cost_type" : "locomotion_with_impact",
        "quad_ctrl_cost_coeff" : 1e-5,
        "lin_vel_cost_coeff" : 1.0,
        "vel_deviation_cost_coeff" : 100.0,
        "quad_impact_cost_coeff" : 1e-5,
        "restingpos_cost_coeff" : 1e-3,
        "clip_impact_cost" : 10.0,        
        "obs_type" : "state+feats+dcom",
        "done_type" : "heads_up_done",
        "quad_ctrl_cost": None,
        "allowed_state_ranges" : {
            "head" : (1.5, 2.0),
        },
        "height_idx" : 2,
        "frame_skip": 1,
        "initializer": None
    },
    "icml_humanoid" : {
        "load_from_dot_in_file": True,
        "world_file" : "icml-humanoid.xml", # will be loaded from .xml.in file!
        "has_ground" : True,
        "experiment" : "walking", # can be "walking" or "standing"
        "cost_type" : "locomotion_with_impact",
        "quad_ctrl_cost_coeff" : 1e-5,
        "lin_vel_cost_coeff" : 1.0,
        "feet_cost_coeff": 1e-2,
        "alive_bonus_coeff": 0.2,
        "stepsize": 0,#1.0,
        "both_feet_cost_coeff": 0,#1.0,
        "vel_deviation_cost_coeff" : 100.0,
        "quad_impact_cost_coeff" : 1e-5,
        "constraint_cost_coeff" : 1e-5,
        "clip_constraint_cost" : 0.5,
        "restingpos_cost_coeff" : 0.0,
        "clip_impact_cost" : 0.5,
        "obs_type" : "state+feats+dcom",
        "done_type" : "state_ranges",
        "quad_ctrl_cost": "[1.0;1.0;1.0;1.0;1.0;1.0;1.0;1.0;1.0;1.0]", # TODO: figure out a proper way to do this (command line comma parsing)        
        "allowed_state_ranges" : {
            2 : (0.9, 2.0),
        },
        "height_idx" : 2,
        "frame_skip": 1,
        "initializer": lambda x, bending: humanoid_initializer(x, bending),
        "knee_bending": 0,#-0.3,
        "forward_init_parameter": 0,#0.5,
        "both_feet_on_ground_cost": 1.0,
    },
    "icml_humanoid_tracking" : {
        "load_from_dot_in_file": True,
        "world_file" : "icml-humanoid-tracking.xml", # will be loaded from .xml.in file!
        "has_ground" : True,
        "experiment" : "walking", # can be "walking" or "standing"
        "cost_type" : "tracking_locomotion",
        "quad_ctrl_cost_coeff" : 1e-5,
        "tracking_vel_cost_coeff" : 1.0,
        "feet_cost_coeff": 1e-2,
        "stepsize": 1.0,
        "both_feet_cost_coeff": 0,#1.0,
        "vel_deviation_cost_coeff" : 100.0,
        "quad_impact_cost_coeff" : 1e-5,
        "constraint_cost_coeff" : 1e-5,
        "clip_constraint_cost" : 0.5,
        "restingpos_cost_coeff" : 0.0,
        "clip_impact_cost" : 10.0,        
        "obs_type" : "state+feats+dcom",
        "done_type" : "state_ranges",
        "quad_ctrl_cost": "[1.0;1.0;1.0;1.0;1.0;1.0;1.0;1.0;1.0;1.0]", # TODO: figure out a proper way to do this (command line comma parsing)        
        "allowed_state_ranges" : {
            4 : (0.9, 2.0),
        },
        "height_idx" : 4,
        "frame_skip": 1,
        "initializer": lambda x, bending: humanoid_initializer_tracking(x, bending),
        "knee_bending": 0,#-0.3,
        "forward_init_parameter": 0,#0.5,
        "both_feet_on_ground_cost": 0,#1.0,
    },
    "3d_humanoid_amputated" : {
        "load_from_dot_in_file": True,
        "world_file" : "humanoid-amputated.xml", # will be loaded from .xml.in file!
        "has_ground" : True,
        "experiment" : "walking", # can be "walking" or "standing"
        "cost_type" : "locomotion_with_impact",
        "quad_ctrl_cost_coeff" : 1e-5,
        "lin_vel_cost_coeff" : 1.0,
        "feet_cost_coeff": 1e-2,
        "stepsize": 1.0,
        "both_feet_cost_coeff": 1.0,
        "vel_deviation_cost_coeff" : 100.0,
        "quad_impact_cost_coeff" : 1e-5,
        "constraint_cost_coeff" : 1e-5,
        "clip_constraint_cost" : 0.5,
        "restingpos_cost_coeff" : 1e-3,
        "clip_impact_cost" : 10.0,        
        "obs_type" : "state+feats+dcom",
        "done_type" : "state_ranges",
        "quad_ctrl_cost": "[2.5;3.0;2.5;0.15;0.15;0.15;0.125;0.3;0.3;0.15;0.15;0.15;0.125;0.3;0.3;2.5;2.5;2.5;2.5;2.5;2.5]", # TODO: figure out a proper way to do this (command line comma parsing)        
        "allowed_state_ranges" : {
            2 : (0.9, 2.0),
        },
        "height_idx" : 2,
        "frame_skip": 1,
        "initializer": lambda x, bending: humanoid_initializer(x, bending),
        "knee_bending": -0.3,
        "forward_init_parameter": 0.5,
        "both_feet_on_ground_cost": 1.0,
    },
    "icml_humanoid_jumper" : {
        "load_from_dot_in_file": True,
        "world_file" : "icml-humanoid.xml", # will be loaded from .xml.in file!
        "has_ground" : True,
        "experiment" : "walking", # can be "walking" or "standing"
        "cost_type" : "locomotion_with_jumping",
        "quad_ctrl_cost_coeff" : 1e-5,
        "lin_vel_cost_coeff" : 1.0,
        "feet_cost_coeff": 1e-2,
        "stepsize": 1.0,
        "both_feet_cost_coeff": 1.0,
        "vel_deviation_cost_coeff" : 100.0,
        "quad_impact_cost_coeff" : 1e-5,
        "constraint_cost_coeff" : 1e-5,
        "clip_constraint_cost" : 0.5,
        "restingpos_cost_coeff" : 1e-3,
        "clip_impact_cost" : 10.0,        
        "obs_type" : "state+feats+dcom",
        "done_type" : "state_ranges",
        "quad_ctrl_cost": "[1.0;1.0;1.0;1.0;1.0;1.0;1.0;1.0;1.0;1.0]", # TODO: figure out a proper way to do this (command line comma parsing)        
        "jump_cost_coeff": 1.0,
        "allowed_state_ranges" : {
            2 : (0.9, 2.0),
        },
        "height_idx" : 2,
        "distance_idx": 1,
        "sand_begin_pos": 5.0,
        "frame_skip": 1,
        "initializer": lambda x, bending: humanoid_initializer(x, bending),
        "knee_bending": -0.3,
        "forward_init_parameter": 0.5,
        "both_feet_on_ground_cost": 1.0,
    },
    "3d_humanoid" : {
        "load_from_dot_in_file": True,
        "world_file" : "humanoid.xml", # will be loaded from .xml.in file!
        "has_ground" : True,
        "experiment" : "walking", # can be "walking" or "standing"
        "cost_type" : "locomotion_with_impact_and_restingpos",
        "quad_ctrl_cost_coeff" : 1e-5,
        "lin_vel_cost_coeff" : 1.0,
        "feet_cost_coeff": 1e-2,
        "stepsize": 1.0,
        "both_feet_cost_coeff": 1.0,
        "vel_deviation_cost_coeff" : 100.0,
        "quad_impact_cost_coeff" : 1e-5,
        "constraint_cost_coeff" : 1e-5,
        "clip_constraint_cost" : 0.5,
        "restingpos_cost_coeff" : 1e-3,
        "clip_impact_cost" : 10.0,        
        "obs_type" : "state+feats+dcom",
        "done_type" : "state_ranges",
        "quad_ctrl_cost": "[2.5;3.0;2.5;0.15;0.15;0.15;0.125;0.3;0.3;0.15;0.15;0.15;0.125;0.3;0.3;2.5;2.5;2.5;2.5;2.5;2.5]", # TODO: figure out a proper way to do this (command line comma parsing)        
        "allowed_state_ranges" : {
            2 : (0.9, 2.0),
        },
        "height_idx" : 2,
        "frame_skip": 1,
        "initializer": lambda x, bending: humanoid_initializer(x, bending),
        "knee_bending": -0.3,
        "forward_init_parameter": 0.5,
        "both_feet_on_ground_cost": 1.0,
    },
    "3d_humanoid_sit" : {
        "load_from_dot_in_file": True,
        "world_file" : "humanoid-sitting.xml", # will be loaded from .xml.in file!
        "has_ground" : True,
        "experiment" : "walking", # can be "walking" or "standing"
        "cost_type" : "sitting_cost",
        "sitting_cost_coeff" : 1.0,
        "stand_up_cost_coeff" : 1.0,
        "quad_ctrl_cost_coeff" : 1e-5,
        "lin_vel_cost_coeff" : 1.0,
        "feet_cost_coeff": 1e-2,
        "stepsize": 1.0,
        "getup_vel": 1.0,
        "both_feet_cost_coeff": 1.0,
        "vel_deviation_cost_coeff" : 100.0,
        "quad_impact_cost_coeff" : 1e-5,
        "constraint_cost_coeff" : 1e-5,
        "clip_constraint_cost" : 0.5,
        "restingpos_cost_coeff" : 1e-3,
        "clip_impact_cost" : 10.0,        
        "obs_type" : "state+feats+dcom",
        "done_type" : "state_ranges",
        "quad_ctrl_cost": None,
        "allowed_state_ranges" : {
            2 : (-100, 100),
        },
        "height_idx" : 2,
        "frame_skip": 1,
        "initializer": lambda x, bending: sitting_initializer(x, bending),
        "knee_bending": -0.3,
        "forward_init_parameter": 0.5,
        "both_feet_on_ground_cost": 1.0,
        "reference" : np.array([[ 0.,     0.,     0.,   ],
            [ 2.69,   0.406,  0.087],
            [ 2.281,  0.363,  0.131],
            [ 2.032,  0.373,  0.099],
            [ 1.647,  0.233,  0.081],
            [ 1.395,  0.215,  0.123],
            [ 2.118,  0.197,  0.163],
            [ 1.769,  0.003,  0.098],
            [ 1.538, -0.131,  0.137],
            [ 2.655,  0.57,   0.233],
            [ 2.638,  0.494,  0.544],
            [ 2.705,  0.101,  0.065],
            [ 2.693,  0.069,  0.136]])
    },
    "3d_humanoid_stand" : {
        "load_from_dot_in_file": True,
        "world_file" : "humanoid-standing.xml", # will be loaded from .xml.in file!
        "has_ground" : True,
        "experiment" : "walking", # can be "walking" or "standing"
        "cost_type" : "standing_cost",
        "standing_cost_coeff" : 1.0,
        "staying_cost_coeff" : 0.0,
        "quad_ctrl_cost_coeff" : 1e-5,
        "lin_vel_cost_coeff" : 1.0,
        "feet_cost_coeff": 1e-2,
        "stepsize": 1.0,
        "getup_vel": 1.0,
        "both_feet_cost_coeff": 1.0,
        "vel_deviation_cost_coeff" : 100.0,
        "quad_impact_cost_coeff" : 1e-5,
        "constraint_cost_coeff" : 1e-5,
        "clip_constraint_cost" : 0.5,
        "restingpos_cost_coeff" : 1e-3,
        "clip_impact_cost" : 10.0,        
        "obs_type" : "state+feats+dcom",
        "done_type" : "state_ranges",
        "quad_ctrl_cost": None,
        "allowed_state_ranges" : {
            2 : (-100, 100),
        },
        "height_idx" : 2,
        "frame_skip": 1,
        "initializer": lambda x, bending: standing_initializer(x, bending),
        "knee_bending": -0.3,
        "forward_init_parameter": 0.5,
        "both_feet_on_ground_cost": 1.0,
    },
    "3d_humanoid_stand_and_walk" : {
        "load_from_dot_in_file": True,
        "world_file" : "humanoid-standing.xml", # will be loaded from .xml.in file!
        "has_ground" : True,
        "experiment" : "walking", # can be "walking" or "standing"
        # "cost_type" : "standing_and_walking_cost",
        "cost_type" : "standing_and_walking_cost_delayed",
        "standing_cost_coeff" : 1.0,
        "staying_cost_coeff" : 0.0,
        "quad_ctrl_cost_coeff" : 1e-5,
        "lin_vel_cost_coeff" : 1.0,
        "quad_vel_cost_coeff" : 1.0,
        "feet_cost_coeff": 1e-2,
        "stepsize": 1.0,
        "getup_vel": 1.0,
        "both_feet_cost_coeff": 1.0,
        "vel_deviation_cost_coeff" : 100.0,
        "quad_impact_cost_coeff" : 1e-5,
        "constraint_cost_coeff" : 1e-5,
        "clip_constraint_cost" : 0.5,
        "restingpos_cost_coeff" : 1e-3,
        "clip_impact_cost" : 10.0,        
        "obs_type" : "state+feats+dcom",
        "done_type" : "state_ranges",
        "quad_ctrl_cost": None,
        "allowed_state_ranges" : {
            2 : (-100, 100),
        },
        "height_idx" : 2,
        "frame_skip": 1,
        "initializer": lambda x, bending: standing_initializer(x, bending),
        "knee_bending": -0.3,
        "forward_init_parameter": 0.5,
        "both_feet_on_ground_cost": 1.0,
    },
    "human_walking" : {
        "load_from_dot_in_file": True,
        "world_file" : "human_walking.xml", # will be loaded from .xml.in file!
        "has_ground" : True,
        "experiment" : "walking", # can be "walking" or "standing"
        "cost_type" : "locomotion_with_impact_and_restingpos",
        "quad_ctrl_cost_coeff" : 1e-5,
        "lin_vel_cost_coeff" : 1.0,
        "feet_cost_coeff": 1e-2,
        "stepsize": 1.0,
        "both_feet_cost_coeff": 1.0,
        "vel_deviation_cost_coeff" : 100.0,
        "quad_impact_cost_coeff" : 1e-5,
        "constraint_cost_coeff" : 1e-5,
        "clip_constraint_cost" : 0.5,
        "restingpos_cost_coeff" : 0.0,
        "clip_impact_cost" : 10.0,        
        "obs_type" : "state+feats+dcom",
        "done_type" : "state_ranges",
        "quad_ctrl_cost": None, # TODO: figure out a proper way to do this (command line comma parsing)        
        "allowed_state_ranges" : {
            # 2 : (0.9, 2.0),
            0 : (-0.5, 1.0)
            # 2 : (-1.0, 1.0)
        },
        "height_idx" : 2,
        "frame_skip": 1,
        "initializer": lambda x, bending: humanoid_initializer(x, bending),
        "knee_bending": -0.3,
        "forward_init_parameter": 0.5,
        "both_feet_on_ground_cost": 1.0,
    },
    "swimmer" : {
        "world_file" : "swimmer.xml",
        "has_ground" : False,
        "cost_type" : "locomotion",
        "quad_ctrl_cost_coeff" : 1e-6,
        "quad_ctrl_cost" : None,
        "lin_vel_cost_coeff" : 1.0,
        "obs_type" : "state",
        "done_type" : "none",
    },
    "hopper" : {
        "world_file" : "hopper.xml",
        "has_ground" : True,
        "cost_type" : "locomotion_with_impact",
        "quad_ctrl_cost_coeff" : 1e-6,
        "lin_vel_cost_coeff" : 1.0,
        "quad_impact_cost_coeff" : 0,        
        "clip_impact_cost" : 10.0,                
        "obs_type" : "state+feats+dcom",
        "done_type" : "state_ranges",
        "quad_ctrl_cost": None,
        "allowed_state_ranges" : {
            0 : (0.8, 2.0),
            2 : (-1.0, 1.0),
        },
        "height_idx" : 0,        
        "frame_skip" : 1,
    },
    "walker2d" : {
        "world_file" : "walker2d.xml",
        "has_ground" : True,
        "cost_type" : "locomotion_with_impact",
        "experiment" : "walking", # can be "walking" or "standing"
        "quad_ctrl_cost_coeff" : 1e-6,
        "lin_vel_cost_coeff" : 1.0,
        "quad_impact_cost_coeff" : 1e-6,        
        "clip_impact_cost" : 10.0,                
        "obs_type" : "state+feats+dcom",
        "done_type" : "state_ranges",
        "quad_ctrl_cost": None,
        "allowed_state_ranges" : {
            0 : (0.8, 2.0),
            2 : (-1.0, 1.0),
        },
        "height_idx" : 0,        
        "frame_skip" : 1,
        "initializer" : None        
    }, 
    "musclewalker2d" : {
        "load_from_dot_in_file": True,
        "world_file" : "simplemusclewalker2d.xml",
        "has_ground" : True,
        "cost_type" : "locomotion_with_impact",
        "experiment" : "walking", # can be "walking" or "standing"
        "quad_ctrl_cost_coeff" : 1e-6,
        "lin_vel_cost_coeff" : 1.0,
        "quad_impact_cost_coeff" : 1e-6,        
        "clip_impact_cost" : 10.0,                
        "obs_type" : "state+feats+dcom",
        "done_type" : "state_ranges",
        "quad_ctrl_cost": None,
        "allowed_state_ranges" : {
            0 : (0.8, 2.0),
            2 : (-1.0, 1.0),
        },
        "height_idx" : 0,        
        "frame_skip" : 1,
        "initializer" : None        
    },
    "igorwalker2d" : {
        "world_file" : "humanoid2Dforward.xml",
        "has_ground" : True,
        "cost_type" : "locomotion_with_impact",
        "quad_ctrl_cost_coeff" : 1e-6,
        "lin_vel_cost_coeff" : 1.0,
        "quad_impact_cost_coeff" : 1e-5,        
        "clip_impact_cost" : 10.0,                
        "obs_type" : "state+feats+dcom",
        "done_type" : "state_ranges",
        "allowed_state_ranges" : {
            1 : (-0.1, 0.5),
            2 : (-0.75, 0.75)
        },
        "height_idx" : 1, 
        "frame_skip" : 4,        
    },        
    "ball_hopper" : {
        "world_file" : "hopper-ball.xml",
        "has_ground" : True,
        "cost_type" : "ball_hopper_cost",
        "quad_ctrl_cost_coeff" : 1e-6,
        "lin_vel_cost_coeff" : 1.0,
        "quad_impact_cost_coeff" : 0,        
        "ball_cost_coeff" : 1.0,
        "ball_pos" : 1.0,
        "clip_impact_cost" : 10.0,                
        "obs_type" : "state+feats+dcom",
        "done_type" : "state_ranges",
        "allowed_state_ranges" : {
            1 : (0.8, 2.0),
            3 : (-1.0, 1.0),
        },
        "height_idx" : 0,        
        "frame_skip" : 1,
        "initializer": None,
        "quad_ctrl_cost": None
    },
}

import re

def do_substitution(in_lines, kws):
    lines_iter = iter(in_lines)
    defn_lines = []
    while True:
        try:
            line = lines_iter.next()
        except StopIteration:
            raise RuntimeError("didn't find line starting with ---")
        if line.startswith('---'):
            break
        else:
            defn_lines.append(line)
    d = {}
    exec("\n".join(defn_lines), d)
    d.update(kws)
    pat = re.compile("\$\((.+?)\)")
    out_lines = []
    for line in lines_iter:
        matches = pat.finditer(line)
        for m in matches:
            line = line.replace(m.group(0), str(eval(m.group(1),d)))
        out_lines.append(line)
    return out_lines

class MJCMDP(MDP):
    def __init__(self, basename, **kws):
        self.kws = kws
        for key in kws.keys():
            assert key in METADATA["default"].keys() or key in METADATA[basename].keys() or key in ["joint_armature","joint_damping","joint_compliance","joint_timeconst","contact_mindist","contact_compliance","contact_timeconst","contact_friction","tact","tdeact","fmax","ctrl_range"]
        self.basename = basename
        self.properties = METADATA["default"]
        self.properties.update(METADATA[basename])
        self._obs_dim = None
        self.properties.update(kws)
        if 'experiment' in self.properties:
            print "doing experiment ", self.properties['experiment']
        self._setup_world(kws)
        if basename == "3d_humanoid":
            body_names = get_body_names(self.model)
            inv_body_names = {v: k for k, v in body_names.items()}
            self.left_foot = inv_body_names['left_foot']
            self.right_foot = inv_body_names['right_foot']
            geom_names = get_geom_names(self.model)
            inv_geom_names = {v: k for k, v in geom_names.items()}
            self.right_geom = [inv_geom_names['right_foot_cap1'], inv_geom_names['right_foot_cap2']]
            self.left_geom = [inv_geom_names['left_foot_cap1'], inv_geom_names['left_foot_cap2']]
        if basename == "icml_humanoid_jumper":
            body_names = get_body_names(self.model)
            inv_body_names = {v: k for k, v in body_names.items()}
            self.left_foot = inv_body_names['left_foot']
            self.right_foot = inv_body_names['right_foot']       
            geom_names = get_geom_names(self.model)
            inv_geom_names = {v: k for k, v in geom_names.items()}
            self.right_geom = [inv_geom_names['right_foot'], inv_geom_names['right_foot']]
            self.left_geom = [inv_geom_names['left_foot'], inv_geom_names['left_foot']]                
        if basename == "atlas":
            body_names = get_body_names(self.model)
            inv_body_names = {v: k for k, v in body_names.items()}
            self.head_idx = inv_body_names['head']
        # assert(self.properties["cost_type"] == "locomotion_with_impact_and_ball")

        self._include_image = False # XXX

    def __hash__(self):
        import hashlib
        h = hashlib.new("md5")
        h.update(self.basename)
        h.update(str(self.kws))
        # just in case, hash this file and xml file
        fname = self._get_filename()
        if self.properties["load_from_dot_in_file"]: fname += ".in"
        with open(fname,"r") as fh:
            h.update(fh.read())
        with open(__file__,"r") as fh:
            h.update(fh.read())
        print h.hexdigest()
        return int(h.hexdigest(),16)



    def _setup_world(self, kws):
        filename = self._prepare_xml_file(kws)
        self.world = mjcpy2.MJCWorld2(filename)
        self.world.SetNumSteps(self.properties["frame_skip"])
        self.world.SetFeats(self.properties["features"])
        self.model = self.world.GetModel()
        self.option = self.world.GetOption()
        self.option.update(kws)
        self.world.SetOption(self.option)

    def _prepare_xml_file(self, kws):
        if self.properties["load_from_dot_in_file"]:
            print "Loading the mjc2 world from .xml.in"
            filename = self._get_filename() + ".in"
            import tempfile
            temp = tempfile.NamedTemporaryFile(delete=False,suffix=".xml")
            with open(filename,"r") as f:
                in_lines = f.readlines()
                out_lines = do_substitution(in_lines, kws)
                temp.writelines(out_lines)
            return temp.name
        else:
            print "Loading the mjc2 world from plain old .xml"
            return self._get_filename()

    def _get_filename(self):
        return osp.join(CTRL_ROOT,"domain_data/mujoco_worlds",self.properties["world_file"])

    def call(self, input_arrs):
        x = input_arrs["x"].ravel()
        u = input_arrs["u"].ravel()

        contacts = self.world.GetContacts(x.astype('float64'))

        combefore = self.world.GetCOMMulti(x.astype('float64').reshape(1,-1))[0]
        y64=x.astype('float64')
        for i in range(self.properties["frame_skip"]):
            y64,feats64 = self.world.Step(y64,u.astype('float64'))
        comafter = self.world.GetCOMMulti(y64.reshape(1,-1))[0]
        y = y64.astype(floatX)
        feats = feats64.astype(floatX)

        obs_type = self.properties["obs_type"]
        if obs_type == "state+feats+dcom":
            if self.basename == "icml_humanoid_jumper":
                o = np.concatenate([y,feats,combefore.astype(floatX)])
            else:
                o = np.concatenate([y,feats,(comafter-combefore).astype(floatX)])
        elif obs_type == "state":
            o = y
        else:
            raise NotImplementedError

        cost = np.array(self._cost(x, u, y, feats, comafter-combefore, contacts)).reshape(1,-1)
        done = self.trial_done(y) or (self.basename == "icml_humanoid_jumper"  and cost.flatten()[-1] < -1e-5)

        if self._include_image:
            o = np.concatenate([o,self.get_img().ravel()])

        return {
            "x" : y.reshape(1,-1),
            "o" : o.reshape(1,-1),
            "c" : cost,
            "done" : done
        }

    def get_img(self):
        import warnings
        warnings.warn("define me")
        return np.zeros((1,)+self.img_shape(),floatX)

    def trial_done(self,x):
        done_type = self.properties["done_type"]
        if done_type == "none":
            return False
        elif done_type == "state_ranges" or done_type == "heads_up_done" or done_type == "state_ranges_ant":
            allowed_ranges = self.properties["allowed_state_ranges"] 
            inds = allowed_ranges.keys()
            lo,hi = np.array(allowed_ranges.values()).T
            if done_type == "state_ranges":
                xi = x[inds]
            elif done_type == "state_ranges_ant":
                D = self.world.GetData()
                xi = D["xipos"][2][inds]
            elif done_type == "heads_up_done":
                xipos = self.world.GetData()["xipos"]
                xi = xipos[self.head_idx, 2]
            return ((xi < lo) | (xi > hi)).any()
        else:
            raise NotImplementedError("invalid done_type %s"%done_type)

    def initialize_mdp_arrays(self):
        x = np.concatenate([self.model["qpos0"].flatten(), np.zeros(self.model["nv"])])
        if self.basename == "ball_hopper":
            x[0] = self.properties["ball_pos"]
        if "initializer" in self.properties and self.properties["initializer"] is not None:
            self.properties["initializer"](x, self.properties)
        if self.properties["cost_type"] == "locomotion_with_impact_and_ball":
             x[0] = 10 * (np.random.rand() - 0.5)
             x[1] = 10 * (np.random.rand() - 0.5)
        u = np.zeros((self.ctrl_dim()),floatX)
        res = self.call({"x": x, "u": u})
        return {"x": res["x"].astype(floatX), "o": res["o"].astype(floatX)}

    def _quad_ctrl_cost(self,u):
        if self.properties.get("quad_ctrl_cost") is None:
            return (.5*self.properties["quad_ctrl_cost_coeff"])*u.dot(u)
        else:
            Q = np.diag(eval(self.properties["quad_ctrl_cost"].replace(";", ",")))
            return (.5*self.properties["quad_ctrl_cost_coeff"])*u.dot(Q.dot(u))

    def _lin_vel_cost(self,dcom):
        return (-self.properties["lin_vel_cost_coeff"]/self.option["timestep"]/self.properties["frame_skip"]) * dcom[0]

    def _feet_cost(self, f):
        xipos = self.world.GetData()["xipos"]
        return self.properties["feet_cost_coeff"]/self.option["timestep"]/self.properties["frame_skip"] * (xipos[self.left_foot,1]**2 + xipos[self.right_foot,1]**2)

    def _vel_dev_cost(self,dcom):
        return self.properties["vel_deviation_cost_coeff"] * (dcom[1]**2 + dcom[2]**2)

    def _quad_impact_cost(self,f):
        startidx = 0
        for (name,size) in self.world.GetFeatDesc():
            if name == "cfrc_ext":
                impact = f[startidx:startidx+size]
                # print "using %s for impact cost"%name
                break
            elif name == "qfrc_impulse":
                impact = f[startidx:startidx+size]
                # print "using %s for impact cost"%name
                break
            else:
                startidx += size
        else:
            raise RuntimeError("need cfrc_ext or qfrc_impact in observations")
        cost = (.5*self.properties["quad_impact_cost_coeff"]*impact.dot(impact)).sum()
        clip_level = self.properties["clip_impact_cost"]
        if clip_level == 0:
            return cost
        else:
            return np.clip(cost, 0, clip_level)

    def _constraint_cost(self, f):
        startidx = 0
        for (name,size) in self.world.GetFeatDesc():
            if name == "qfrc_constraint":
                constr = f[startidx:startidx+size]
                break
            else:
                startidx += size
        else:
            raise RuntimeError("need qfrc_constraint in observations")
        cost = (.5*self.properties["constraint_cost_coeff"]*constr.dot(constr)).sum()
        clip_level = self.properties["clip_constraint_cost"]
        if clip_level == 0:
            return cost
        else:
            return np.clip(cost, 0, clip_level)      

    def _restingpos_cost(self, x):
        a = -0.25
        if self.properties["experiment"] == "walking":
            # resting position of left and right knee should be a
            return self.properties["restingpos_cost_coeff"] * (x[15]**2 + x[21]**2 + x[7]**2 + x[8]**2 + x[9]**2 + (x[22] + 1.0)**2 + (x[23] - 0.5)**2 + (x[24] + 1.0)**2 + (x[25] - 1.0)**2 + (x[26] + 0.5)**2 + (x[27 + 1.0])**2)
        elif self.properties["experiment"] == "standing":
            return self.properties["restingpos_cost_coeff"] * (x[13] - a) ** 2
        else:
            raise NotImplementedError

    def _ball_cost(self,x,f):
        features = extract_feature(self.world, f, "cvel")
        return -1.0 * self.properties["ball_cost_coeff"] * features[9]

    def _upright_cost(self,x):
        return self.properties["upright_cost_coeff"] * ((x[8] - x[9])**2 + x[10]**2)

    def _find_ball_cost(self, ball, torso):
        return self.properties["find_ball_cost_coeff"] * ((ball[0] - torso[0])**2 + (ball[1] - torso[1])**2)



    def _cost(self, x, u, y, f, dcom, contacts): #pylint: disable=W0613
        cost_type = self.properties["cost_type"]
        if cost_type == "locomotion_with_impact":
            alive_bonus = x[0]*0-self.properties["alive_bonus_coeff"]
            return np.array([ self._lin_vel_cost(dcom) , self._quad_ctrl_cost(u) , self._quad_impact_cost(f), alive_bonus ] ,floatX)
        if cost_type == "tracking_locomotion":
            alive_bonus = x[0]*0-self.properties["alive_bonus_coeff"]
            return np.array([ -self.properties["tracking_vel_cost_coeff"]*(x[21] * x[0] + x[22] * x[1]) , self._quad_ctrl_cost(u) , self._quad_impact_cost(f), alive_bonus ] ,floatX)
        if cost_type == "locomotion_with_impact_and_ball":    
            D = self.world.GetData()
            alive_bonus = x[0]*0-self.properties["alive_bonus_coeff"]
            return np.array([ self._find_ball_cost(D["xipos"][1], D["xipos"][2]) , self._quad_ctrl_cost(u) , self._quad_impact_cost(f), alive_bonus ] ,floatX)
        elif cost_type == "standing_cost":
            # m = self.world.GetModel()
            # print x[28+2]
            D = self.world.GetData()
            # return np.array([self.properties["standing_cost_coeff"] * (x[28+2] - self.properties["getup_vel"])**2, self._quad_ctrl_cost(u)])
            return np.array([self.properties["standing_cost_coeff"] * (D["xipos"][1][2] - 1.43)**2, self.properties["staying_cost_coeff"] * (D["xipos"][1][0]**2 + D["xipos"][1][1]**2), self._quad_ctrl_cost(u), self._quad_impact_cost(f)])
        elif cost_type == "standing_and_walking_cost":
            D = self.world.GetData()
            return np.array([self.properties["standing_cost_coeff"] * (D["xipos"][1][2] - 1.43)**2, self.properties["quad_vel_cost_coeff"]*(x[28] + 1.5)**2, self._quad_ctrl_cost(u), self._quad_impact_cost(f)])
        elif cost_type == "standing_and_walking_cost_delayed":
            D = self.world.GetData()
            return np.array([self.properties["standing_cost_coeff"] * (D["xipos"][1][2] - 1.43)**2, -self._lin_vel_cost(dcom) * (D["xipos"][1][2] > 1.2), self._quad_ctrl_cost(u), self._quad_impact_cost(f)])
        elif cost_type == "sitting_cost":
            # x[2] = 1.05
            # x[12] = -1.5
            # x[18] = -1.5
            # x[13] = -1.5
            # x[19] = -1.5
            D = self.world.GetData()
            
            return np.array([self.properties["sitting_cost_coeff"] * np.linalg.norm(self.properties["reference"] - D["xipos"])**2, self.properties["stand_up_cost_coeff"] * (D["xipos"][1][2] - 1.43)**2])
        elif cost_type == "locomotion_with_impact_tars":
            alive_bonus = x[0]*0-self.properties["alive_bonus_coeff"]
            return np.array([ self._lin_vel_cost(dcom) , self._quad_ctrl_cost(u) , self._quad_impact_cost(f), self._upright_cost(f), alive_bonus ] ,floatX)
        elif cost_type == "locomotion_with_impact_and_restingpos":
            alive_bonus = x[0]*0-self.properties["alive_bonus_coeff"]
            return np.array([ self._lin_vel_cost(dcom) , self._quad_ctrl_cost(u) , self._quad_impact_cost(f), alive_bonus, self._restingpos_cost(x) ] ,floatX)
        elif cost_type == "locomotion_with_jumping":
            # TODO: Inefficient
            jump_cost = 0.0
            xipos = self.world.GetData()["xipos"]
            left_foot_pos = xipos[self.left_foot,0]
            right_foot_pos = xipos[self.right_foot,0]
            left_foot_on_ground = False
            right_foot_on_ground = False
            xipos = self.world.GetData()["xipos"]
            first = map(lambda x: x[1], contacts)
            second = map(lambda x: x[2], contacts)
            bodies_in_contact = first + second
            if len([y for y in bodies_in_contact if y in self.left_geom]):
                # left_foot_on_ground
                jump_cost = left_foot_pos - self.properties["sand_begin_pos"]
            if len([y for y in bodies_in_contact if y in self.right_geom]):
                # right_foot_on_ground
                jump_cost = right_foot_pos - self.properties["sand_begin_pos"]
            if jump_cost < 0.0:
                jump_cost = 0.0
            alive_bonus = x[0]*0-self.properties["alive_bonus_coeff"]
            return np.array([ self._lin_vel_cost(dcom) , self._quad_ctrl_cost(u) , self._quad_impact_cost(f), alive_bonus, self._restingpos_cost(x) , -self.properties["jump_cost_coeff"]*jump_cost] ,floatX)
        elif cost_type == "locomotion":
            return np.array([ self._lin_vel_cost(dcom) , self._quad_ctrl_cost(u)],floatX)
        elif cost_type == "ball_hopper_cost":
            alive_bonus = x[0]*0-self.properties["alive_bonus_coeff"]
            return np.array([ self._lin_vel_cost(dcom) , self._quad_ctrl_cost(u) , self._quad_impact_cost(f), alive_bonus, self._ball_cost(x,f) ] ,floatX)
        else:
            raise NotImplementedError("unrecognized cost type %s"%self.properties["cost_type"])

    def unscaled_cost(self, cost_dict):
        return {"unscaled_vel_cost": cost_dict["vel"]/self.properties["lin_vel_cost_coeff"]}

    def cost_names(self):
        cost_type = self.properties["cost_type"]
        if cost_type == "standing_cost":
            return ["torso_height", "staying", "ctrl", "impact"]
        if cost_type == "standing_and_walking_cost":
            return ["torso_height", "vel", "ctrl", "impact"]
        if cost_type == "standing_and_walking_cost_delayed":
            return ["torso_height", "vel", "ctrl", "impact"]
        if cost_type == "sitting_cost":
            return ["final_pos", "stand_up"]
        if cost_type == "tracking_locomotion":
            return ["tracking","ctrl","impact","notdone"]
        if cost_type == "locomotion_with_impact":
            return ["vel","ctrl","impact","notdone"]
        if cost_type == "locomotion_with_impact_and_ball":
            return ["find_ball","ctrl","impact","notdone"]
        elif cost_type == "locomotion_with_impact_tars":
            return ["vel", "ctrl", "impact", "upright", "notdone"]
        elif cost_type == "locomotion_with_impact_and_restingpos":
            return ["vel", "ctrl","impact", "notdone", "restingpos"]
        elif cost_type == "locomotion_with_jumping":
            return ["vel", "ctrl","impact", "notdone", "restingpos", "jumping"]
        elif cost_type == "locomotion":
            return ["vel","ctrl"]
        elif cost_type == "ball_hopper_cost":
            return ["vel","ctrl","impact","notdone","ball"]

    def input_info(self):
        return {
            "x" : (self.state_dim(),floatX),
            "u" : (self.ctrl_dim(),floatX),
            "c" : (self.num_costs(),floatX)
        }    

    def output_info(self):
        return {
            "x" : (self.state_dim(),floatX),
            "o" : (self.obs_dim(),floatX),
            "img" : (self.img_shape(),floatX), 
            "c" : (self.num_costs(),floatX),
            "done" : (None,'uint8')
        }

    def plot(self, input_arrs):
        x = input_arrs["x"]
        assert x.ndim==2 and x.shape[0]==1
        self.world.Plot(x[0].astype('float64'))

    def state_dim(self):
        return self.model["nq"] + self.model["nv"]

    def obs_dim(self):
        if self._obs_dim is None:
            self._obs_dim = self.initialize_mdp_arrays()["o"].size
        return self._obs_dim

    def ctrl_desc(self):
        return self.model["nu"]

    def ctrl_dim(self):
        return self.model["nu"]

    def ctrl_bounds(self):
        return self.model["actuator_ctrlrange"].T.astype(floatX)

    def obs_ranges(self):
        raise NotImplementedError

    def obs_names_sizes(self):
        """
        Description of the observation of the form
        [(name1, dim1), (name2, dim2), ...]
        """
        obs_type = self.properties["obs_type"]
        if obs_type == "state+feats+dcom":
            return [("state",self.state_dim())] + list(self.world.GetFeatDesc()) + [("dcom",3)]
        elif obs_type == "state":
            return [("state",self.state_dim()), ("dcom",3)]
        else:
            raise NotImplementedError

    def img_shape(self):
        return (3,30,30)




def main():    
    mdp = MJCMDP("3d_humanoid")
    mdp.validate()
    d = mdp.world.GetModel()
    x = np.concatenate([d["qpos0"].flatten(), np.zeros(d["nv"])])
    print "x", x.dtype
    u = np.ones(d["nu"])
    result = mdp.call({"x": x, "u": u})
    print result

    #from control4.algs.save_load_utils import construct_agent
    #from control4.core.rollout import rollout
    #agent = construct_agent({"agent_module": "control4.agents.random_continuous_agent"},mdp)
    #max_steps = 10000
    #result = rollout(mdp,agent,max_steps,save_arrs=("o",))
    #obs = result[1]["o"] #pylint: disable=W0612

    # print np.array(obs).squeeze().shape
    # print mdp.obs_dim()

    # print map(lambda x: x.std(), obs)
    # print map(lambda x: x.max(), obs)

    # print np.array(map(lambda o: np.percentile(o, [5,95]), obs))



if __name__ == "__main__":
    main()
