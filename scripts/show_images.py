from scripts.export_image_data import extract_images
import argparse
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pyprind
import itertools


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
    fig = plt.figure()
    print 'Generating matplotlib image object...'
    for i in pyprind.prog_bar(range(len(image_data))):
        imgs.append([[i])])
    print 'Saving animation'
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    ani = animation.ArtistAnimation(fig, imgs, interval=66)
    ani.save(args.output, writer=writer)
