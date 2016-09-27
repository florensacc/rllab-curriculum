from sandbox.rocky.neural_learner.doom_utils.textmap import *
from io import StringIO
import numpy as np
import random

# random.seed(0)
# np.random.seed(0)

player = Thing(x=0, y=0, type=1, id=1)
# this is the goal the player should reach
# some candidate things:
# 5 - blue keycard (kind of too small)

bluecard = Thing(x=0, y=0, z=20, type=5, id=2)

textmap = Textmap(namespace="zdoom", items=[])

things = [player, bluecard]



# maze_gen = DFSGridMazeGenerator()
# maze = maze_gen.gen_maze(n_row=5, n_col=5)

# maze = np.asarray([
#     [0, 0, 0, 0, 0],
#     [0, 1, 0, 1, 0],
#     [0, 1, 0, 1, 0],
#     [0, 1, 1, 1, 0],
#     [0, 0, 0, 0, 0],
# ])
maze = np.asarray([
    [0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
])

linedefs = []
sidedefs = [Sidedef(sector=0, texturemiddle="BRICK9")]
sectors = []
vertices = []

for i in range(maze.shape[0]):
    for j in range(maze.shape[1]):
        if maze[i][j] == 1:  # if not wall
            vs = [
                (96 * i, 96 * (j + 1)),
                (96 * (i + 1), 96 * (j + 1)),
                (96 * (i + 1), 96 * j),
                (96 * i, 96 * j)
            ]
            for v in vs:
                if v not in vertices:
                    vertices.append(v)
            ls = []
            for vid in range(4):
                vid_next = (vid + 1) % 4
                ls.append((vertices.index(vs[vid]), vertices.index(vs[vid_next])))
            for l in ls:
                if l not in linedefs:
                    linedefs.append(l)

sectors.append(
    Sector(
        texturefloor="FLAT3",
        textureceiling="FLAT2",
        heightceiling=104,
        lightlevel=210,
    )
)

linedefs = [l for l in linedefs if (l[1], l[0]) not in linedefs]

textmap.items.extend(things)
textmap.items.extend([Vertex(x=v[0], y=v[1]) for v in vertices])
textmap.items.extend([Linedef(v1=l[0], v2=l[1], blocking=True, sidefront=0) for l in linedefs])
textmap.items.extend(sidedefs)
textmap.items.extend(sectors)

for thing in things:
    while True:
        x = np.random.randint(0, maze.shape[0] * 96)
        y = np.random.randint(0, maze.shape[1] * 96)
        i = x // 96
        j = y // 96
        if maze[i][j] == 1:
            thing.x = x
            thing.y = y
            break

sio = StringIO()
textmap.write(sio)
print(sio.getvalue())
