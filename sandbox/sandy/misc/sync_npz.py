#!/usr/bin/env python

""" One-off script that uploads .pkl.np.z files from EC2 machines to S3
"""

import os.path as osp
import subprocess

EC2_MACHINES = [
        "ec2-54-149-231-219.us-west-2.compute.amazonaws.com",
        "ec2-54-202-85-245.us-west-2.compute.amazonaws.com",
        "ec2-54-201-73-5.us-west-2.compute.amazonaws.com",
        "ec2-54-202-47-109.us-west-2.compute.amazonaws.com",
        "ec2-54-202-87-195.us-west-2.compute.amazonaws.com",
        "ec2-54-202-74-186.us-west-2.compute.amazonaws.com",
        "ec2-54-202-75-12.us-west-2.compute.amazonaws.com",
        "ec2-54-202-79-196.us-west-2.compute.amazonaws.com",
        "ec2-54-201-148-206.us-west-2.compute.amazonaws.com",
        "ec2-54-202-53-188.us-west-2.compute.amazonaws.com",
        "ec2-54-202-74-246.us-west-2.compute.amazonaws.com",
        "ec2-54-202-83-247.us-west-2.compute.amazonaws.com",
        "ec2-54-187-176-101.us-west-2.compute.amazonaws.com",
        "ec2-54-202-50-130.us-west-2.compute.amazonaws.com",
        "ec2-54-201-216-18.us-west-2.compute.amazonaws.com",
        "ec2-54-202-57-143.us-west-2.compute.amazonaws.com",
        "ec2-54-202-81-112.us-west-2.compute.amazonaws.com",
        "ec2-54-202-47-199.us-west-2.compute.amazonaws.com",
        "ec2-54-202-57-75.us-west-2.compute.amazonaws.com",
        "ec2-54-202-78-138.us-west-2.compute.amazonaws.com",
        ]

EXP_DIRS = [
        "exp035b_20170123_024351_863480_space_invaders",
        "exp035b_20170123_024351_351975_space_invaders",
        "exp035b_20170123_024350_860780_space_invaders",
        "exp035b_20170123_024350_301768_space_invaders",
        "exp035b_20170123_024349_700808_space_invaders",
        "exp035b_20170123_024349_215243_seaquest",
        "exp035b_20170123_024348_537082_seaquest",
        "exp035b_20170123_024348_013145_seaquest",
        "exp035b_20170123_024347_518899_seaquest",
        "exp035b_20170123_024346_968980_seaquest",
        "exp035b_20170123_024346_458235_pong",
        "exp035b_20170123_024345_954386_pong",
        "exp035b_20170123_024345_450999_pong",
        "exp035b_20170123_024344_929254_pong",
        "exp035b_20170123_024344_435868_pong",
        "exp035b_20170123_024343_938358_chopper_command",
        "exp035b_20170123_024343_420026_chopper_command",
        "exp035b_20170123_024342_925898_chopper_command",
        "exp035b_20170123_024342_336928_chopper_command",
        "exp035b_20170123_024337_748991_chopper_command",
        ]

ALGO = 'deep-q-rl'
GAMES = ['space', 'seaquest', 'pong', 'chopper']
EC2_BASE_DIR = '/home/shhuang/src/rllab-private/data/local/' + ALGO
S3_BASE_DIR = 's3://rllab-shhuang/rllab/experiments/' + ALGO

SSH_KEY = '/home/shhuang/.ssh/shhuang-us-west-2.pem'
SYNC_S3_SCRIPT = '/home/shhuang/src/rllab-private/scripts/sync_s3.py'

def get_game(exp_dir):
    for g in GAMES:
        if g in exp_dir:
            return g
    raise NotImplementedError

def main():
    for i, host in enumerate(EC2_MACHINES):
        exp_dir = EXP_DIRS[i]
        exp_index = exp_dir.split('_')[0]
        exp_name = exp_index + '-' + get_game(exp_dir)
        ec2_dir = osp.join(EC2_BASE_DIR, exp_name)
        s3_dir = osp.join(S3_BASE_DIR, exp_name)

        cmd = 'aws s3 sync ' + ec2_dir + ' ' + s3_dir + ' --region us-west-2'
        print("Executing:", cmd)

        ssh = subprocess.Popen(["ssh", "-o", "StrictHostKeyChecking no",
                                "-i", SSH_KEY, 'ubuntu@%s' % host, cmd],
                               shell=False,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
        result = ssh.stdout.readlines()
        if result == []:
            error = ssh.stderr.readlines()
            print("ERROR: %s" % error)
        else:
            print("Result:", result)
        sync_cmd = ['python', SYNC_S3_SCRIPT, ALGO + '/' + exp_name]
        print("Executing:", ' '.join(sync_cmd))
        subprocess.Popen(sync_cmd,
                         shell=False,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)

        #ssh -i "shhuang-us-west-2.pem" root@ec2-54-202-47-109.us-west-2.compute.amazonaws.com
        #aws s3 sync ec2_dir s3_dir --region us-west-2

if __name__ == "__main__":
    main()
