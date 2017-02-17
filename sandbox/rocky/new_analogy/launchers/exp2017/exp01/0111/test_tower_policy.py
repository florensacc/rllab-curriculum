from gpr_package.bin import tower_copter_policy as tower


task_id = tower.get_task_from_text("ab")
expr = tower.Experiment(2, 1000)
env = expr.make(task_id)


tower.PolicyRunner(env, tower.CopterPolicy(task_id)).run()

