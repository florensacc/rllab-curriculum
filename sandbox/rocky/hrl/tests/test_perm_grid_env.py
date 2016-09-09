


from nose2.tools import such
from sandbox.rocky.hrl.envs.perm_grid_env import PermGridEnv
import numpy as np

with such.A("Perm Grid Env") as it:
    @it.should
    def test_env():
        env = PermGridEnv(size=5, n_objects=5, random_restart=False)
        perm = np.random.permutation(np.arange(5))

        path = env.generate_training_path((0, 0), perm)
        env.agent_pos = (0, 0)
        env.visit_order = perm
        env.n_visited = 0
        score = 0
        for a in path["actions"]:
            _, rew, _, _ = env.step(env.action_space.unflatten(a))
            score += rew
        it.assertEqual(score, 5)

it.createTests(globals())
