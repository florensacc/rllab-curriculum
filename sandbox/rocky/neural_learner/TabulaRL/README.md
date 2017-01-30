# TabulaRL

Simple tabular reinforcement learning in Python.

This is a very simple project for tabular reinforcement learning (TabulaRL) that I wrote during my PhD studies.
There are already several great open source packages for reinforcement learning, including:

- [RL Glue](http://glue.rl-community.org/wiki/Main_Page)
- [RLPY](https://github.com/rlpy/rlpy)
- [PyBrain](http://pybrain.org/)
- [OpenAI Gym](https://gym.openai.com/)
- [Microsoft minecraft](http://www.pcworld.com/article/3043895/analytics/microsoft-to-open-source-ai-development-platform-based-on-minecraft.html) (to come)

Out of these, it looks like [OpenAI Gym](https://gym.openai.com/) is currently the most promising... if I was starting this all again maybe I would have focused my attention here!

That said, I found that for studying simple tabular RL problems these frameworks were all a little bit of overkill.
If you are experimenting with small, simple, tabular MDPs or just learning about RL for the first time it can be nice to have a really simple bit of code to get started.
The nice thing about this code instead of something like Gym is:

- All written in simple Python+Numpy, no complicated dependencies beyond standard [Anaconda](https://www.continuum.io/downloads)
- Very few lines of code, potentially more accessible to novice programmers.
- Focus on very simple problems.

But the main reason I'm putting this code up is so that people can more easily implement/verify some of the ideas in some of my papers!

