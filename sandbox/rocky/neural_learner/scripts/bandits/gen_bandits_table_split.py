import csv
import os
from collections import defaultdict
from io import StringIO
import numpy as np
import scipy.stats

from rllab import config

TEMPLATE = r"""
\begin{table}[H]
\begin{center}
\begin{tabular}{llllllllll}
\multicolumn{1}{c}{\bf Setup}
&\multicolumn{1}{c}{\bf Random}
&\multicolumn{1}{c}{\bf Gittins}
&\multicolumn{1}{c}{\bf TS}
&\multicolumn{1}{c}{\bf OTS}
&\multicolumn{1}{c}{\bf UCB1}
&\multicolumn{1}{c}{\bf UCB1$^*$}
&\multicolumn{1}{c}{\bf $\epsilon$-Greedy}
&\multicolumn{1}{c}{\bf Greedy}
&\multicolumn{1}{c}{\bf RL$^2$}
\\ \hline \\
%s
\end{tabular}
\end{center}
\end{table}
"""

selected_settings = [
    (0, 0.01),
    (0.01, 0.05),
    (0.05, 0.1),
    (0.1, 0.3),
    (0.3, 0.5),
    (0.5, 1.0),
    # (0, 1.0),
]

files = [
    "random.csv",
    "gittins.csv",
    "ts.csv",
    "ots.csv",
    "ucb.csv",
    "ucb_default.csv",
    "egreedy.csv",
    "greedy.csv",
    "rnn.csv",
]

results = {k: [] for k in selected_settings}  # [dict() for _ in range(len(files))]

# results = defaultdict(list)

folder = os.path.join(config.PROJECT_PATH, "data/iclr2016_prereview")
# files = os.listdir(folder)
for idx, file in enumerate(files):
    path = os.path.join(folder, file)
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for line in reader:
            # import ipdb; ipdb.set_trace()
            n_arms = int(line['n_arms'])
            n_episodes = int(line['n_episodes'])

            key = (float(line['epsilon_from']), float(line['epsilon_to']))

            if key in results:
                mean = float(line['best_arm_percent'])
                std = np.std(int(mean*1000) * [1] + int((1-mean)*1000)*[0]) / np.sqrt(1000 -1)
                # import ipdb; ipdb.set_trace()
                results[key].append((mean, std))#float(line['avg']), float(line['stdev']), float(line[
                                                                                                  # 'best_arm_percent'])))

n_trials = 1000

buf = StringIO()

for epsilon_from, epsilon_to in sorted(selected_settings, key=lambda x: x[1]):
    buf.write(
        r"$\epsilon \in [{}, {}]$".format(str(epsilon_from), str(epsilon_to))
    )
    buf.write(" & ")

    key = (epsilon_from, epsilon_to)

    best_mean = max([mean for mean, _ in results[key]])
    best_std = [std for mean, std in results[key] if mean == best_mean][0]

    components = []

    for mean, std in results[key]:
        # perform a t-test
        result = scipy.stats.ttest_ind_from_stats(best_mean, best_std * np.sqrt(n_trials - 1), n_trials, mean,
                                                  std * np.sqrt(n_trials - 1), n_trials, equal_var=False)
        # if significantly worse
        if 1 - result.pvalue / 2 < 0.95 or mean >= best_mean:
            components.append("$\\bf %.1f\\%%$" % (mean*100))#, std))
        else:
            components.append("$%.1f\\%%$" % (mean*100))#, std))


        # elif mean > best_mean:

        # import ipdb;
        # ipdb.set_trace()

    buf.write(" & ".join(components))
    buf.write("\\\\\n")

print(TEMPLATE % buf.getvalue())
