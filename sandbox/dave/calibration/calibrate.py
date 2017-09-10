import sys
import os
from shutil import copyfile
#from fetch_setup.logger import get_logger
import datetime
import click
import numpy as np
from sandbox.dave.calibration.calibration_utils import load_calibration_data, evaluate_xml, calibrate_xml
from sandbox.dave.calibration.xml_utils import set_joint_params
from sandbox.dave.calibration.calibration_config import default_xml, default_xml_path
from rllab.misc.instrument import stub
from rllab.misc.instrument import VariantGenerator
# import copy
stub(globals())


all_joints = ["l_shoulder_pan_joint", \
                  "l_shoulder_lift_joint", \
                  "l_upper_arm_roll_joint", \
                  "l_elbow_flex_joint", \
                  "l_forearm_roll_joint", \
                  "l_wrist_flex_joint", \
                  "l_wrist_roll_joint", \
                  ]

all_params = ['joint_damping', \
                  'joint_frictionloss', \
                  'joint_armature', \
                  'joint_stiffness', \
              ]


@click.command()
@click.option('--data_path', default='/home/young_clgan/GitRepos/rllab-private/sandbox/dave/calibration/calibration',
              help='directory from which to pull fetch data')
@click.option('--evaluate_only', is_flag=False, default=True,
              help='only evaluate the current .xml model, no optimization performed')
@click.option('--joint', default="None", help='joint to calibrate')

# class VG(VariantGenerator):
#
#     @variant
#     def joint_damping(self):
#         return [0.1, 1, 5, 10, 25, 50, 100]
#
#     @variant
#     def joint_frictionloss(self):
#         return [0.05, 0.1, 0.25, 0.5, 1, 1.25]
#
#     @variant
#     def joint_armature(self):
#         return [0.05, 0.1, 0.5, 1, 2.5, 5, 7.5]
#
#     @variant
#     def joint_stiffness(self):
#         return [0.05, 0.1, 0.5, 1, 2.5, 5, 7.5]
#
# @variant
# def joint(self):
#     return all_joints
#



def main(data_path, evaluate_only, joint):
    best_result = 10000
    vg = VariantGenerator()
    vg.add("joint_damping", [0.1, 1, 5, 10, 25, 50, 100])
    vg.add("joint_frictionloss", [0.05, 0.1, 0.25, 0.5, 1, 1.25])
    vg.add("joint_armature", [0.05, 0.1, 0.5, 1, 2.5, 5, 7.5])
    vg.add("joint_stiffness", [0.05, 0.1, 0.5, 1, 2.5, 5, 7.5])
    v = vg.variants()

    path_to_xml = '/home/young_clgan/GitRepos/rllab-private/sandbox/dave/vendor/mujoco_models/pr2_lego_calibration.xml'
    best_xml_path = '/home/young_clgan/GitRepos/rllab-private/sandbox/dave/vendor/mujoco_models/pr2_lego_calibration_best.xml'
    # param_updates = {joint: dict([('joint_damping', v['joint_damping']),
    #                               ('joint_frictionloss', v['joint_frictionloss']),
    #                               ('joint_armature', v['joint_armature']),
    #                               ('joint_stiffness', v['joint_stiffness'])
    #                               ])}

    param_updates = [v['joint_damping'],
                     v['joint_frictionloss'],
                     v['joint_armature'],
                     v['joint_stiffness'],
                                  ]


    # evaluate_only = True
    params = dict([(j, all_params) for j in all_joints])
    train_data, test_data = load_calibration_data(all_joints, data_path)
    # import pdb; pdb.set_trace()
    current_result = evaluate_xml(test_data)
    # aprox_curr = copy.copy(current_result)
    # if evaluate_only:
    #     del aprox_curr["l_forearm_roll_joint"]
    #     del aprox_curr["l_wrist_roll_joint"]
    # print('Best_new result: ', best_result, "   |    Current result", np.mean(list(aprox_curr.values())))
    # if np.mean(list(aprox_curr.values())) < best_result:
    #     best_result = np.mean(list(aprox_curr.values()))
    #     print('best_new result: ', best_result)
    #     copyfile(default_xml_path, best_xml_path)
    if not evaluate_only:
        optimized_params = calibrate_xml(train_data, params)
        new_result = evaluate_xml(train_data, param_updates=optimized_params)
        print("Optimized params: %s", str(optimized_params))
    result_str = '\n\n'
    result_str += '  Current xml score: {0:6.3f} %\n'.format(100.*np.mean(list(current_result.values())))
    if not evaluate_only:
        result_str += '  New xml score: {0:6.3f} %\n'.format(100.*np.mean(list(new_result.values())))
    result_str += '\n  Detailed results (current xml): \n'
    for joint in all_joints:
        result_str += '    {0}: {1:6.3f} %\n'.format(joint, 100.*current_result[joint])
    if not evaluate_only:
        result_str += '\n  Detailed results (new xml): \n'
        for joint in all_joints:
            result_str += '    {0}: {1:6.3f} %\n'.format(joint, 100.*new_result[joint])

    if not evaluate_only:
        thresh = 0.0001 # Must be 0.01% better
        if np.mean(list(new_result.values())) < np.mean(list(current_result.values())) - thresh:
            default_xml_dir = '/'.join(str.split(default_xml_path, '/')[:-1])
            try:
                os.mkdir(os.path.join(default_xml_dir, 'backup'))
            except OSError:
                pass
            backup_file = os.path.join(default_xml_dir, 'backup',
                                       'main_' + '_'.join(str(datetime.datetime.now()).split(' ')) + '.xml')
            print('Copying existing xml to file %s', backup_file)
            copyfile(default_xml_path, backup_file)
            print('Writing result to file %s', default_xml_path)
            new_xml = set_joint_params(default_xml, optimized_params)
            # with open(default_xml_path, 'w') as f:
            #     f.write(new_xml)
            # if np.mean(list(new_result.values())) < best_result:
            #     best_result = np.mean(list(new_result.values()))
            #     print(best_result)
            #     with open(best_xml_path, 'w') as f:
            #         f.seek(0)
            #         f.write(new_xml)
            #         f.truncate()
            #         f.close()

        else:
            if np.mean(list(current_result.values())) < best_result:
                best_result = np.mean(list(current_result.values()))
                print(best_result)
                copyfile(default_xml_path, best_xml_path)
            print('xml worsened. not saving')

if __name__ == '__main__':
  main()
