# rllab-curriculum

This codebase is self-contained to reproduce the results in:
- [Carlos Florensa, David Held, Xinyang Geng, Pieter Abbeel. *Automatic Goal Generation for Reinforcement Learning Agents*. In Proceedings of the 35th International Conference on Machine Learning (ICML) 2018](http://proceedings.mlr.press/v80/florensa18a.html).
- [Carlos Florensa, David Held, Markus Wulfmeier, Michael Zhang, Pieter Abbeel. *Reverse Curriculum Generation for Reinforcement Learning*. In Conference on Robot Learning (CoRL) 2017](http://proceedings.mlr.press/v78/florensa17a.html).

To setup `rllab`, please see documentation at [https://rllab.readthedocs.org/en/latest/](https://rllab.readthedocs.org/en/latest/).

## Goal Generation
To run the maze-ant goal experiments, run:

`python curriculum/experiments/goals/maze_ant/maze_ant_gan.py`

In the same directory are all the files to lauch all the baselines presented in the [*Automatic Goal Generation for RL Agents*](http://proceedings.mlr.press/v80/florensa18a.html) paper, and more. The performances obtained should match the figure found in 

`data/Figures/maze_ant/maze_ant_baselines_long.png`

## Reverse Curriculum
To run the key-hole manipulation experiments, run:

`python curriculum/experiments/starts/arm3d/arm3d_key/arm3d_key_brownian.py`

In the same directory are all the files to lauch all the baselines presented in the [*Reverse Curriculum Generation for RL*](http://proceedings.mlr.press/v78/florensa17a.html) paper, and more. The performances obtained should match the figure found in 

`data/Figures/arm3d-key/main.png`
