# Code by Erik Sweet and Bill Basener

import random
import numpy as np
import scipy
import scipy.io as sio
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import pyprind
from sandbox.rocky.neural_planner.gridworld_benchmark import to_sparse_adj_graph
from rllab import config
import os.path as osp


NUM_ROWS = 3#14#4#3
NUM_COLS = 3#14#4#3
N_MAPS = 1000#45000
STATE_BATCH_SIZE = 10
SEED = 11

SHAPE = (NUM_ROWS * 2 + 1, NUM_COLS * 2 + 1)

EXP_DATA_PATH = osp.join(config.PROJECT_PATH, "sandbox/rocky/exp_data")

random.seed(SEED)
np.random.seed(SEED)

maps = []
all_from_xs = []
all_from_ys = []
all_to_xs = []
all_to_ys = []
all_actions = []

print("Generating data...")

for idx in pyprind.prog_bar(range(N_MAPS)):
    print(idx)

    num_rows = NUM_ROWS
    num_cols = NUM_COLS

    M = np.zeros((num_rows, num_cols, 5), dtype=np.uint8)
    # The array M is going to hold the array information for each cell.
    # The first four coordinates tell if walls exist on those sides
    # and the fifth indicates if the cell has been visited in the search.
    # M(LEFT, UP, RIGHT, DOWN, CHECK_IF_VISITED)
    # image = np.zeros((num_rows * 10, num_cols * 10), dtype=np.uint8)
    # The array image is going to be the output image to display

    # Set starting row and column
    r = 0
    c = 0
    history = [(r, c)]#, (0, 0), (0, c), (r, 0)]  # The history is the

    # Trace a path though the cells of the maze and open walls along the path.
    # We do this with a while loop, repeating the loop until there is no history,
    # which would mean we backtracked to the initial start.
    while history:
        M[r, c, 4] = 1  # designate this location as visited
        # check if the adjacent cells are valid for moving to
        check = []
        if c > 0 and M[r, c - 1, 4] == 0:
            check.append('L')
        if r > 0 and M[r - 1, c, 4] == 0:
            check.append('U')
        if c < num_cols - 1 and M[r, c + 1, 4] == 0:
            check.append('R')
        if r < num_rows - 1 and M[r + 1, c, 4] == 0:
            check.append('D')

        if len(check):  # If there is a valid cell to move to.
            # Mark the walls between cells as open if we move
            history.append([r, c])
            move_direction = random.choice(check)
            if move_direction == 'L':
                M[r, c, 0] = 1
                c = c - 1
                M[r, c, 2] = 1
            if move_direction == 'U':
                M[r, c, 1] = 1
                r = r - 1
                M[r, c, 3] = 1
            if move_direction == 'R':
                M[r, c, 2] = 1
                c = c + 1
                M[r, c, 0] = 1
            if move_direction == 'D':
                M[r, c, 3] = 1
                r = r + 1
                M[r, c, 1] = 1
        else:  # If there are no valid cells to move to.
            # retrace one step back in history if no move is possible
            r, c = history.pop()

    # Open the walls at the start and finish
    M[0, 0, 0] = 1
    # M[0, num_cols - 1, 1] = 1
    M[num_rows - 1, num_cols - 1, 2] = 1
    # M[num_rows - 1, 0, 3] = 1

    grid = np.zeros((num_rows * 2 + 1, num_cols * 2 + 1))

    # Generate the image for display
    for row in range(0, num_rows):
        for col in range(0, num_cols):
            cell_data = M[row, col]
            grid[row * 2 + 1, col * 2 + 1] = 1
            if cell_data[0] == 1:
                grid[row * 2 + 1, col * 2] = 1
            if cell_data[1] == 1:
                grid[row * 2, col * 2 + 1] = 1
            if cell_data[2] == 1:
                grid[row * 2 + 1, col * 2 + 2] = 1
            if cell_data[3] == 1:
                grid[row * 2 + 2, col * 2 + 1] = 1
            # for i in range(10 * row + 1, 10 * row + 9):
            #     image[i, range(10 * col + 1, 10 * col + 9)] = 255
            #     if cell_data[0] == 1:
            #         image[range(10 * row + 1, 10 * row + 9), 10 * col] = 255
            #     if cell_data[1] == 1:
            #         image[10 * row, range(10 * col + 1, 10 * col + 9)] = 255
            #     if cell_data[2] == 1:
            #         image[range(10 * row + 1, 10 * row + 9), 10 * col + 9] = 255
            #     if cell_data[3] == 1:
            #         image[10 * row + 9, range(10 * col + 1, 10 * col + 9)] = 255

    # Display the image
    # plt.imshow(image, cmap=cm.Greys_r, interpolation='none')
    plt.imshow(grid, cmap=cm.Greys_r, interpolation='nearest')
    plt.show()

    maps.append(grid)

    grid = 1 - grid

    graph = to_sparse_adj_graph(grid)
    to_x, to_y = random.choice(list(zip(*np.where(grid == 0))))
    to_id = to_x * SHAPE[1] + to_y
    # sample random reachable point on the map
    sps, preds = scipy.sparse.csgraph.dijkstra(graph, directed=False, indices=[to_id], return_predecessors=True)
    from_ids = np.where(np.logical_and(np.logical_not(np.isinf(sps)), np.arange(sps.shape[1]) != to_id))[1]
    if len(from_ids) < STATE_BATCH_SIZE * 2:
        continue

    from_ids = np.cast['int'](np.random.choice(from_ids, size=STATE_BATCH_SIZE, replace=False))
    from_xs = from_ids // SHAPE[1]
    from_ys = from_ids % SHAPE[1]

    next_ids = np.cast['int'](preds[0, from_ids])

    # Compute the desired action
    actions = -np.ones_like(from_ids)
    # Top
    actions[next_ids == from_ids - SHAPE[1]] = 0
    # Down
    actions[next_ids == from_ids + SHAPE[1]] = 1
    # Right
    actions[next_ids == from_ids + 1] = 2
    # Left
    actions[next_ids == from_ids - 1] = 3
    # Top right
    actions[next_ids == from_ids - SHAPE[1] + 1] = 4
    # Top left
    actions[next_ids == from_ids - SHAPE[1] - 1] = 5
    # Bottom right
    actions[next_ids == from_ids + SHAPE[1] + 1] = 6
    # Bottom left
    actions[next_ids == from_ids + SHAPE[1] - 1] = 7

    try:
        assert np.all(np.greater_equal(actions, 0))
    except AssertionError:
        import ipdb;

        ipdb.set_trace()

    all_from_xs.append(from_xs)
    all_from_ys.append(from_ys)
    all_to_xs.append(to_x)
    all_to_ys.append(to_y)
    all_actions.append(actions)


batch_im_data = np.cast['uint8'](maps) * 255
state_x_data = np.asarray(all_from_xs)
state_y_data = np.asarray(all_from_ys)
batch_label_data = np.asarray(all_actions)
batch_value_data = np.zeros_like(batch_im_data)
batch_value_data[np.arange(len(maps)), all_to_xs, all_to_ys] = 10  # why 10???

print("Saving data...")
sio.savemat(
    osp.join(EXP_DATA_PATH, "gridworld_%d.mat" % SHAPE[0]),
    dict(
        batch_im_data=batch_im_data,
        state_x_data=state_x_data,
        state_y_data=state_y_data,
        batch_label_data=batch_label_data,
        batch_value_data=batch_value_data,
    ),
    do_compression=True,
)
