
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

"""
Iterates through folders from peg experiments and makes plots
"""

parser = argparse.ArgumentParser()
parser.add_argument('file', type=str,
                    help='path to the directory with states generated')

# data = np.load("buffer_states.pkl")
# print("Length of data: {}".format(len(data)))
#
# coordinate = True
# fig, ax = plt.subplots()
# for i in range(len(data)):
#     plt.scatter(data[i][-2], data[i][-1], c="blue")
# plt.xlim([-0.02, 0.02])
# plt.ylim([-0.02, 0.02])
# plt.title("Generated peg positions")
# plt.show()
# fig.savefig("peg.png")
#
# fig, ax = plt.subplots()
# for i in range(len(data)):
#     plt.scatter(range(9), data[i], c="lightgreen")
#     plt.scatter(range(9), (0.387, 1.137, -2.028, -1.744, 2.029, -0.873, 1.55, 0, 0), c="black")
# plt.title("Variation for each joint")
# plt.show()
# fig.savefig("joint_variations.png")
