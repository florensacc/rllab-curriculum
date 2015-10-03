#!/usr/bin/python

from mdp import AtariMDP
from ale_python_interface import ALEInterface, ale_lib
import argparse
import sys
import numpy as np
import pyprind
import itertools

def extract_images(states, rom_path, as_np=False):
    ale_lib.setLoggerLevelError()
    ale = ALEInterface()
    ale.loadROM(rom_path)
    print 'Recovering images...'
    N = states.shape[0]
    if as_np:
        image_data = np.zeros((N,) + AtariMDP.to_rgb(ale).shape)
    else:
        image_data = [None] * N
    for i in pyprind.prog_bar(range(N)):
        ale.load_serialized(states[i,:])
        ale.act(0)
        img = AtariMDP.to_rgb(ale)
        image_data[i] = img[:,:,2::-1]
    return image_data



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export image data from snapshot files. This only saves the rgb data to a file instead of generating images.')
    parser.add_argument('-i', '--input', metavar='INPUT_FILE', type=str, help='input file', required=True)
    parser.add_argument('-r', '--rom', metavar='ROM_PATH', type=str, help='rom file', required=True)
    parser.add_argument('-o', '--output', metavar='OUTPUT_FILE', type=str, help='output file', required=True)
    args = parser.parse_args()
    data = np.load(args.input)
    if 'all_states' in data:
        image_data = extract_images(data['all_states'].astype(np.uint8), args.rom, as_np=False)
    elif 'image_data' in data:
        image_data = data['image_data']
    else:
        print 'Invalid data format: must either have all_states or image_data'
        sys.exit(1)
    import cv2
    import cv2.cv as cv
    height, width, _ = image_data[0].shape
    writer = cv2.VideoWriter(args.output, cv.CV_FOURCC('D', 'I', 'V', 'X'), 15, (width, height))
    print 'Exporting video...'
    for idx, image in enumerate(image_data):
        writer.write(image.astype(np.uint8))
