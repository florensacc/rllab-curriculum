from gpr_package.bin import tower_copter_policy as tower


def to_profile():
    for idx in range(1):
        print(idx)
        task_id = tower.get_task_from_text("ab")
        expr = tower.Experiment(2, 1000)
        env = expr.make(task_id)

if __name__ == "__main__":
    to_profile()
