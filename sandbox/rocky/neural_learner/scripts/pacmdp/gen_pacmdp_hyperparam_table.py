import csv
import os
from io import StringIO
import numpy as np

from rllab import config

selected_settings = [
    (10),
    (25),
    (50),
    (75),
    (100),
]

files = [
    ("optimistic_psrl_mdp.csv", "OPSRL"),
    ("beb_mdp.csv", "BEB"),
    ("epsilon_greedy_mdp.csv", "$\epsilon$-Greedy"),
    ("ucrl2_mdp.csv", "UCRL2"),
]

results = {k: [] for k in selected_settings}

for idx, (file, name) in enumerate(files):
    path = os.path.join(config.PROJECT_PATH, "data/iclr2016_full/%s" % file)

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
            n_episodes = int(line['n_episodes'])

            key = (n_episodes)

            if key in results:

                if 'best_n_samp' in line:
                    param = line['best_n_samp']
                elif 'best_scaling' in line:
                    param = "%.6f" % float(line['best_scaling'])
                elif 'best_epsilon' in line:
                    param = line['best_epsilon']
                else:
                    import ipdb;

                    ipdb.set_trace()

                to_write.append((n_episodes, param))

    to_write = sorted(to_write, key=lambda x: (x[0]))

    for n_episodes, param in to_write:
        buf.write(r"$n={n}$ & ".format(n=n_episodes))
        buf.write(r"${param}$ & \\".format(param=param))
        buf.write("\n")

    buf.write(r"\cline{1-3}")
    buf.write("\n")
    buf.write(r"\end{tabular}")
    buf.write("\n")
    buf.write(r"\end{table}")
    buf.write("\n")

    print(buf.getvalue())
