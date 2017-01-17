#!/usr/bin/env python

""" Produces video of side-by-side original, noise, and adversarial images
"""

import argparse
import cv2
import h5py
import numpy as np
import os
import subprocess

WHITE = np.array([255,255,255])

def obs_to_rgb(obs):
    # Assumes obs is scaled to be from -1 to 1
    obs = (obs + 1.0) / 2.0
    obs = obs[:,:,np.newaxis]*WHITE
    return np.uint8(np.around(obs))

def visualize_adversary(rollouts_file, output_dir, output_prefix, \
                        frames_per_sec=20, pad=5, writing=30, max_timestep=10000):
    adv_rollouts_f = h5py.File(rollouts_file, 'r')
    obs_h, obs_w = adv_rollouts_f['rollouts']['0']['orig_input'].shape
    img_h = obs_h + pad*2
    img_w = obs_w*3 + pad*4

    for i in range(len(adv_rollouts_f['rollouts'])):
        if i > max_timestep:
            break
        if i % 1000 == 0:
            print("At timestep", i)
        g = adv_rollouts_f['rollouts'][str(i)]
        img = WHITE * np.ones((img_h,img_w,3), np.uint8)
        img[pad:pad+obs_h, pad:pad+obs_w] = obs_to_rgb(g['orig_input'][()])
        img[pad:pad+obs_h, pad*2+obs_w:pad*2+obs_w*2] = obs_to_rgb(g['change_unscaled'][()])
        img[pad:pad+obs_h, pad*3+obs_w*2:pad*3+obs_w*3] = obs_to_rgb(g['adv_input'][()])
        
        cv2.imwrite(os.path.join(output_dir, output_prefix + '_{0:06d}.png'.format(i)), img)

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
    adv_rollouts_f.close()
    return os.path.join(output_dir, output_prefix + '_video.mp4')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('rollouts_file')
    args = parser.parse_args()

    output_dir, output_prefix = os.path.split(args.rollouts_file)
    output_prefix = output_prefix.split('.')[0]
    visualize_adversary(args.rollouts_file, output_dir, output_prefix)

if __name__ == "__main__":
    main()
