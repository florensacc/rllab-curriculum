# How to run VIME

Variational Information Maximizing Exploration (VIME) as presented in Curiosity-driven Exploration in Deep Reinforcement Learning via Bayesian Neural Networks by *R. Houthooft, X. Chen, Y. Duan, J. Schulman, F. De Turck, P. Abbeel* (http://arxiv.org/abs/1605.09674). 

Train a Bayesian neural network (BNN) on a simple regression task via `python vime/dynamics/run_bnn.py`.

Execute TRPO+VIME on the hierarchical SwimmerGather environment `python vime/experiments/run_trpo_expl.py`.