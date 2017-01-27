

from gpr_package.bin import tower_copter_policy as tower
from sandbox.rocky.new_analogy.envs.custom_tower import Experiment
from sandbox.rocky.new_analogy.scripts.tower.crippled_policy import CrippledPolicy

task_id = tower.get_task_from_text("ab")
expr = Experiment(2, 250)
env = expr.make(task_id)


tower.PolicyRunner(env, CrippledPolicy(tower.CopterPolicy(task_id))).run()
