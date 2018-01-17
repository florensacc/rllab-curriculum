# rllab-curriculum

For `rllab` setup, please see documentation at [https://rllab.readthedocs.org/en/latest/](https://rllab.readthedocs.org/en/latest/).

## Goal Generation
To run the maze-ant goal experiments, run:

`python curriculum/experiments/goals/maze_ant/maze_ant_gan.py`

In the same directory are all the files to lauch all the baselines presented in the [*Automatic Goal Generation for RL Agents*](https://arxiv.org/abs/1705.06366) paper, and more. The performances obtained should match the figure found in 

`data/Figures/maze_goal/maze_ant_baselines_long.png`

## Reverse Curriculum
To run the key-hole manipulation experiments, run:

`python curriculum/experiments/starts/arm3d/arm3d_key/arm3d_key_brownian.py`

In the same directory are all the files to lauch all the baselines presented in the [*Reverse Curriculum Generation for RL*](https://arxiv.org/pdf/1707.05300.pdf) paper, and more. The performances obtained should match the figure found in 

`data/Figures/arm3d-key/main.png`
