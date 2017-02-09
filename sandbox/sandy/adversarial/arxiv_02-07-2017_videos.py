#!/usr/bin/python

"""Assumes rollouts have already been computed and saved, in the base directory.
   Also assumes there are exactly two h5 files in each output directory, one
   with epsilon = 0.
"""
import cv2, h5py, os, os.path as osp, numpy as np, subprocess
from sandbox.sandy.misc.util import create_dir_if_needed
from sandbox.sandy.adversarial.vis import obs_to_rgb, action_prob_to_rgb, \
                                          PROB_HEIGHT_RATIO, WHITE

BASE_DIR = "/home/shhuang/src/rllab-private/data/local/adv-rollouts/exp012b"
VIDEO_BACKGROUND = "/home/shhuang/src/rllab-private/sandbox/sandy/adversarial/video_text_{norm}.png"
OUTPUT_DIR = "/home/shhuang/src/rllab-private/data/local/adv-rollouts/exp012b/arxiv_videos"
SCALE = 4
OFFSET_Y = 95
PAD = 5
FRAMES_PER_SEC = 20

def create_video(no_adv_h5, adv_h5, background, output_dir, scale=SCALE, pad=PAD, \
                 frames_per_sec=FRAMES_PER_SEC):
    create_dir_if_needed(output_dir)

    scale = int(scale)
    assert scale > 0

    noadv_f = h5py.File(no_adv_h5, 'r')
    adv_f = h5py.File(adv_h5, 'r')
    obs_h, obs_w = adv_f['rollouts']['0']['orig_input'].shape

    obs_h *= scale
    obs_w *= scale
    prob_w = obs_w
    prob_h = int(prob_w / PROB_HEIGHT_RATIO)
    #img_h = obs_h + pad*3 + prob_h
    #img_w = obs_w*3 + pad*4

    adv_path_lengths = adv_f['path_lengths'][()]
    if 'path_lengths' not in noadv_f:
        print("\tSKIPPING")
        return
    noadv_path_lengths = noadv_f['path_lengths'][()]
    assert len(adv_path_lengths) == len(noadv_path_lengths)
    adv_idx = 0
    noadv_idx = 0
    cumul_idx = 0
    last_img = cv2.imread(background)
    img_h, img_w, _ = last_img.shape
    adv_offset_x = img_w - pad*3 - obs_w*3

    for i in range(len(noadv_path_lengths)):
        path_len = max(noadv_path_lengths[i], adv_path_lengths[i])
        print("Num timesteps:", path_len)
        for t in range(path_len):
            img = np.array(last_img)
            if cumul_idx % 100 == 0:
                print("At timestep", cumul_idx)
            if t < noadv_path_lengths[i]:
                noadv_g = noadv_f['rollouts'][str(noadv_idx)]
                img[OFFSET_Y+pad:OFFSET_Y+pad+obs_h, pad:pad+obs_w] = \
                        obs_to_rgb(noadv_g['orig_input'][()], scale=scale)
                img[OFFSET_Y+pad*2+obs_h:OFFSET_Y+pad*2+obs_h+prob_h, pad:pad+obs_w] = \
                        action_prob_to_rgb(noadv_g['action_prob_orig'][()], \
                                           prob_w, prob_h, noadv_f['algo'][()])
                noadv_idx += 1

            if t < adv_path_lengths[i]:
                adv_g = adv_f['rollouts'][str(adv_idx)]
                img[OFFSET_Y+pad:OFFSET_Y+pad+obs_h, adv_offset_x:adv_offset_x+obs_w] = \
                        obs_to_rgb(adv_g['orig_input'][()], scale=scale)
                img[OFFSET_Y+pad:OFFSET_Y+pad+obs_h, adv_offset_x+pad+obs_w:adv_offset_x+pad+obs_w*2] = \
                        obs_to_rgb(adv_g['change_unscaled'][()], scale=scale)
                img[OFFSET_Y+pad:OFFSET_Y+pad+obs_h, adv_offset_x+pad*2+obs_w*2:adv_offset_x+pad*2+obs_w*3] = \
                        obs_to_rgb(adv_g['adv_input'][()], scale=scale)

                img[OFFSET_Y+pad*2+obs_h:OFFSET_Y+pad*2+obs_h+prob_h,
                    adv_offset_x:adv_offset_x+obs_w] = \
                        action_prob_to_rgb(adv_g['action_prob_orig'][()], \
                                           prob_w, prob_h, adv_f['algo'][()])
                img[OFFSET_Y+pad*2+obs_h:OFFSET_Y+pad*2+obs_h+prob_h, \
                     adv_offset_x+pad*2+obs_w*2:adv_offset_x+pad*2+obs_w*3] = \
                         action_prob_to_rgb(adv_g['action_prob_adv'][()], \
                                            prob_w, prob_h, adv_f['algo'][()])
                adv_idx += 1
            last_img = img

            #else:  # Adversarially-perturbed policy's rollout is already
            #       # complete, so show blank frames until non-perturbed policy finishes
            #    img[adv_offset_x:adv_offset_x+obs_h, OFFSET_Y+pad:OFFSET_Y+pad+obs_w] *= 0
            #    img[adv_offset_x:adv_offset_x+obs_h, pad*2+obs_w:pad*2+obs_w*2] *= 0
            #    img[adv_offset_x:adv_offset_x+obs_h, pad*3+obs_w*2:pad*3+obs_w*3] *= 0
            #    img[adv_offset_x+pad+obs_h:adv_offset_x+pad+obs_h+prob_h,
            #        OFFSET_Y+pad:OFFSET_Y+pad+obs_w] *= 0
            #    img[adv_offset_x+pad+obs_h:adv_offset_x+pad+obs_h+prob_h, \
            #         pad*3+obs_w*2:pad*3+obs_w*3] *= 0

            output_prefix = osp.split(adv_h5)[1].split('.')[0]
            cv2.imwrite(os.path.join(output_dir, output_prefix + '_{0:06d}.png'.format(cumul_idx)), img)
            cumul_idx += 1

    subprocess.check_call(['ffmpeg', '-r', str(frames_per_sec), '-f', 'image2', \
                           '-s', str(img_h)+'x'+str(img_w), '-i',
                           os.path.join(output_dir, output_prefix + '_%06d.png'), '-vcodec', \
                           'libx264', '-crf', '0', '-pix_fmt', 'yuv420p', \
                           os.path.join(output_dir, output_prefix + 'video.mp4')])

    # Remove the saved-off .png files used to generate the video
    fnames = os.listdir(output_dir)
    for f in fnames:
        if f.endswith('.png') and output_prefix in f:
            os.remove(os.path.join(output_dir, f))
    noadv_f.close()
    adv_f.close()
    return os.path.join(output_dir, output_prefix + '_cumul_video.mp4')

def main():
    output_dirs = os.listdir(BASE_DIR)
    _, exp_idx = osp.split(BASE_DIR)
    output_dirs = [d for d in output_dirs if d.startswith(exp_idx[:6])]

    for o_dir in output_dirs:
        h5_files = os.listdir(osp.join(BASE_DIR, o_dir))
        h5_files = [f for f in h5_files if f.endswith('.h5') and not 'allvariants' in f]
        if len(h5_files) != 2:
            print(h5_files)
            continue
        assert len(h5_files) == 2
        print(h5_files)
        no_adv_h5 = [f for f in h5_files if '_0_' in f][0]
        adv_h5 = [f for f in h5_files if '_0_' not in f][0]
        norm = adv_h5.split('_')[1]
        background = VIDEO_BACKGROUND.format(norm=norm)

        adv_h5 = osp.join(BASE_DIR, o_dir, adv_h5)
        no_adv_h5 = osp.join(BASE_DIR, o_dir, no_adv_h5)

        create_video(no_adv_h5, adv_h5, background, OUTPUT_DIR)

if __name__ == "__main__":
    main()

