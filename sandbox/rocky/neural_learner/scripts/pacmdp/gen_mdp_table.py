import csv
import os
from io import StringIO
import numpy as np
import scipy.stats


from rllab import config

TEMPLATE = r"""
\begin{table}[th]
\caption{Random MDP Results}
\label{mdp-table}
\begin{center}
\begin{tabular}{lllllllll}
\multicolumn{1}{c}{\bf Setup}
&\multicolumn{1}{c}{\bf Random}
&\multicolumn{1}{c}{\bf PSRL}
&\multicolumn{1}{c}{\bf OPSRL}
&\multicolumn{1}{c}{\bf UCRL2}
&\multicolumn{1}{c}{\bf BEB}
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
    (10),
    (25),
    (50),
    (75),
    (100),
]

files = [
    "random_mdp.csv",
    "psrl_mdp.csv",
    "optimistic_psrl_mdp.csv",
    "ucrl2_mdp.csv",
    "beb_mdp.csv",
    "epsilon_greedy_mdp.csv",
    "greedy_mdp.csv",
    "rnn_mdp.csv",
]

results = {k: [] for k in selected_settings}  # [dict() for _ in range(len(files))]

for idx, file in enumerate(files):
    path = os.path.join(config.PROJECT_PATH, "data/iclr2016_full/%s" % file)
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for line in reader:
            n_episodes = int(line['n_episodes'])

            key = (n_episodes)

            if key in results:
                # results[key].append("$%.2f \pm %.2f$" % (float(line['avg']), float(line['stdev'])))
                results[key].append((float(line['avg']), float(line['stdev'])))  # "$%.2f$" % (float(line['avg'])))

                # import ipdb; ipdb.set_trace()
n_trials = 1000

buf = StringIO()
for T in sorted(selected_settings):
    buf.write(
        "$n={0}$".format(T)
    )
    buf.write(" & ")

    best_mean = max([mean for mean, _ in results[(T)]])
    best_std = [std for mean, std in results[(T)] if mean == best_mean][0]

    components = []

    for mean, std in results[(T)]:
        # perform a t-test
        result = scipy.stats.ttest_ind_from_stats(best_mean, best_std * np.sqrt(n_trials - 1), n_trials, mean,
                                                  std * np.sqrt(n_trials - 1), n_trials, equal_var=False)
        # if significantly worse
        if 1 - result.pvalue / 2 < 0.95 or mean >= best_mean:
            components.append("$\\bf %.1f$" % (mean))#, std))
        else:
            components.append("$%.1f$" % (mean))#, std))


        # elif mean > best_mean:

        # import ipdb;
        # ipdb.set_trace()

    buf.write(" & ".join(components))
    #     [
    #         "$%.2f$" % mean  # if mean != best_mean else "$\\bf %.2f$" % mean
    #         for mean, _ in results[(K, T)]
    #         ]
    # ))
    buf.write("\\\\\n")

print(TEMPLATE % buf.getvalue())


# for K in n_arms:
#
#     TEMPLATE = r"""
# \begin{tikzpicture}
# \begin{axis}[
#     xlabel={Horizon (T)},
#     ylabel={Total reward},
#     xtick={10,50,100,500},
#     legend pos=north west,
#     ymajorgrids=true,
#     grid style=dashed,
# ]
#
# \addplot[
#     color=blue,
#     mark=square,
#     ]
#     coordinates {
#     %s
#     };
#     \addlegendentry{Gittins Index};
# \addplot[
#     color=red,
#     mark=triangle,
#     ]
#     coordinates {
#     %s
#     };
#     \addlegendentry{RNN Policy}
#
# \end{axis}
# \end{tikzpicture}
#     """
#     gittins_coords = "".join(["(%.2f,%.2f)" % (x, y) for x, y in gittins_results[K]])
#     rnn_coords = "".join(["(%.2f,%.2f)" % (x, y) for x, y in rnn_results[K]])
#
#     print(TEMPLATE % (gittins_coords, rnn_coords))
