import csv
import os
from io import StringIO

from rllab import config


selected_settings = [
    (5, 10),
    (10, 10),
    (50, 10),
    (5, 100),
    (10, 100),
    (50, 100),
    (5, 500),
    (10, 500),
    (50, 500),
]

files = [
    ("optimistic_thompson_mab.csv", "OTS"),
    ("ucb1_mab.csv", "UCB1"),
    ("epsilon_greedy_mab.csv", "$\epsilon$-Greedy"),
]

results = {k: [] for k in selected_settings}

for idx, (file, name) in enumerate(files):
    path = os.path.join(config.PROJECT_PATH, "data/iclr2016_new/%s" % file)

    buf = StringIO()

    buf.write(r"\begin{table}[th]")
    buf.write("\n")
    buf.write(r"\centering")
    buf.write("\n")
    buf.write(r"\caption{Best hyperparameter for %s}" % name)
    buf.write("\n")
    buf.write(r"\begin{tabular}{l|ll}")
    buf.write("\n")
    buf.write(r"\cline{1-3} \\ [-8pt]")
    buf.write("\n")
    buf.write(r"")

    to_write = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for line in reader:
            n_arms = int(line['n_arms'])
            n_episodes = int(line['n_episodes'])

            key = (n_arms, n_episodes)

            if key in results:

                if 'best_n_samples' in line:
                    param = line['best_n_samples']
                elif 'best_c' in line:
                    param = line['best_c']
                elif 'best_epsilon' in line:
                    param = line['best_epsilon']
                else:
                    import ipdb; ipdb.set_trace()

                to_write.append((n_episodes, n_arms, param))

    to_write = sorted(to_write, key=lambda x: (x[0], x[1]))

    for n_episodes, n_arms, param in to_write:

        buf.write(r"$n={n}, k={k}$ & ".format(n=n_episodes, k=n_arms))
        buf.write(r"${param}$ & \\".format(param=param))
        buf.write("\n")

    buf.write(r"\cline{1-3}")
    buf.write("\n")
    buf.write(r"\end{tabular}")
    buf.write("\n")
    buf.write(r"\end{table}")
    buf.write("\n")

    print(buf.getvalue())
