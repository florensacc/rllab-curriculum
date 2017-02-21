"""One-off script for getting data from saved videos for website
"""

import os, os.path as osp
import subprocess

VIDEO_DIR = "/home/shhuang/src/rllab-private/data/local/adv-rollouts/exp013/arxiv_videos"
EXP_TO_ALGO = {'exp027': 'trpo', 'exp035c': 'dqn', 'exp036': 'a3c'}

def video_fname_to_info(fname):
    fname = fname[:fname.index("video.mp4")]
    video_info = fname.split('_')
    game = video_info[-1]
    if game == 'command':
        game = 'chopper'
    elif game == 'invaders':
        game = 'space'
    norm = video_info[1]
    eps = float(video_info[2].replace('-', '.', 1))

    policy_adv_idx = fname.index('exp')
    policy_target_idx = fname[policy_adv_idx+1:].index('exp')

    policy_adv_name = fname[policy_adv_idx:policy_adv_idx+policy_target_idx]
    policy_target_name = fname[policy_adv_idx+policy_target_idx+1:]
    transfer = (policy_adv_name != policy_target_name)

    policy_adv_algo = EXP_TO_ALGO[policy_adv_name.split('_')[0]]
    policy_target_algo = EXP_TO_ALGO[policy_target_name.split('_')[0]]

    return (game, norm, eps, policy_adv_algo, policy_target_algo, transfer)

def print_video_info(video_info):
    video_info = sorted(video_info)
    for info in video_info:
        eps_str = str(info[1])
        if eps_str == "3.125e-05":
            eps_str = "0.00003125"
        elif eps_str == "6.25e-05":
            eps_str = "0.0000625"
        print(' '*12 + '\"' + info[0] + '\": [\"' + '\", ' + eps_str + '],')
        cp_out = subprocess.Popen("cp " + osp.join(VIDEO_DIR,info[2]) + " " \
                                        + osp.join(VIDEO_DIR,info[0]+'.mp4'), \
                                  shell=True, stdout=subprocess.PIPE, \
                                  stderr=subprocess.PIPE)
        print("Copying params file:", osp.join(VIDEO_DIR,info[2]), \
              osp.join(VIDEO_DIR,info[0]+'.mp4'), cp_out.stderr.readlines())

def main():
    video_fnames = os.listdir(VIDEO_DIR)
    all_video_info = []
    for fname in video_fnames:
        game, norm, eps, policy_adv_algo, policy_target_algo, transfer = \
                video_fname_to_info(fname)
        if transfer:
            div_id = '_'.join([game, policy_adv_algo, policy_target_algo, norm])
        else:
            assert policy_adv_algo == policy_target_algo
            div_id = '_'.join([game, policy_adv_algo, norm])
        all_video_info.append((div_id, eps, fname))

    print_video_info(all_video_info)

if __name__ == "__main__":
    main()
