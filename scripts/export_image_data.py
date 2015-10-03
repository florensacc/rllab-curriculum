#!/usr/bin/python

import numpy as np
from mdp import AtariMDP
from ale_python_interface import ALEInterface, ale_lib
from struct import unpack
import argparse
import pyprind

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
        image_data[i] = AtariMDP.to_rgb(ale)
    return image_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export image data from snapshot files. This only saves the rgb data to a file instead of generating images.')

    parser.add_argument('-i', '--input', metavar='INPUT_FILE', type=str, help='input file', required=True)
    parser.add_argument('-o', '--output', metavar='OUTPUT_FILE', type=str, help='output file', required=True)
    parser.add_argument('-r', '--rom', metavar='ROM_PATH', type=str, help='rom file', required=True)
    args = parser.parse_args()

    data = np.load(args.input)

    print 'Extracting data...'
    image_data = extract_images(data['all_states'].astype(np.uint8), args.rom, as_np=True)
    print 'Saving image data in compressed format...'
    np.savez_compressed(args.output, **{ 'image_data': image_data })
