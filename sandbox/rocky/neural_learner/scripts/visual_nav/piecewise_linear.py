import numpy as np


def prune_traj(traj):

    pruned_points = []

    cur_points = []

    for idx, pt in enumerate(traj):
        if len(cur_points) <= 1:
            cur_points.append(pt)
            continue

        new_points = np.asarray(cur_points + [pt])
        A = np.vstack([new_points[:, 0], np.ones(len(new_points))]).T
        y = new_points[:, 1]

        residual = np.linalg.lstsq(A, y)[1]
        if len(residual) == 0:
            cur_points.append(pt)
            continue
        elif residual[0] / len(y) > 1:
            pruned_points.append(cur_points[0])
            cur_points = [pt]
        else:
            cur_points.append(pt)
        # import ipdb; ipdb.set_trace()

    if len(cur_points) > 0:
        pruned_points.append(cur_points[0])
        pruned_points.append(cur_points[-1])


    return np.asarray(pruned_points)
    # pass


if __name__ == "__main__":
    origi_traj = np.load("traj.npz")["traj"]
    print(len(origi_traj))
    traj = prune_traj(origi_traj)
    print(len(traj))

    import matplotlib.pyplot as plt

    for pt0, pt1 in zip(origi_traj[:-1], origi_traj[1:]):
        plt.plot([pt0[0], pt1[0]], [pt0[1], pt1[1]], 'b')#traj[:, 0], traj[:, 1], 'ro')

    for pt0, pt1 in zip(traj[:-1], traj[1:]):
        plt.plot([pt0[0], pt1[0]], [pt0[1], pt1[1]], 'r')#traj[:, 0], traj[:, 1], 'ro')
    plt.show()



    # print(traj)
