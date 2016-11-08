import csv
import os
from io import StringIO
import numpy as np
import scipy.stats

from rllab import config

TEMPLATE = r"""
\begin{table}[th]
\caption{\mabcaption}
\label{mab-table}
\begin{center}
\begin{tabular}{lllllllll}
\multicolumn{1}{c}{\bf Setup}
&\multicolumn{1}{c}{\bf Random}
&\multicolumn{1}{c}{\bf Gittins}
&\multicolumn{1}{c}{\bf TS}
&\multicolumn{1}{c}{\bf OTS}
&\multicolumn{1}{c}{\bf UCB1}
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
    "random_mab.csv",
    "approx_gittins_mab.csv",
    "thompson_mab.csv",
    "optimistic_thompson_mab.csv",
    "ucb1_mab.csv",
    "epsilon_greedy_mab.csv",
    "greedy_mab.csv",
    "rnn_mab.csv",
]

results = {k: [] for k in selected_settings}  # [dict() for _ in range(len(files))]


zero_scores = dict()
one_scores = dict()

for idx, file in enumerate(files):
    path = os.path.join(config.PROJECT_PATH, "data/iclr2016/%s" % file)
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for line in reader:
            n_arms = int(line['n_arms'])
            n_episodes = int(line['n_episodes'])

            key = (n_arms, n_episodes)

            if key in results:
                # results[key].append("$%.2f \pm %.2f$" % (float(line['avg']), float(line['stdev'])))
                results[key].append((float(line['avg']), float(line['stdev'])))  # "$%.2f$" % (float(line['avg'])))

                if "random_mab" in file:
                    zero_scores[key] = float(line['avg'])
                elif "gittins" in file:
                    one_scores[key] = float(line['avg'])

                # import ipdb; ipdb.set_trace()
n_trials = 1000

buf = StringIO()
for K, T in sorted(selected_settings, key=lambda x: x[1]):
    buf.write(
        "$n={1}, k={0}$".format(K, T)
    )
    buf.write(" & ")

    key = (K, T)

    best_mean = max([mean for mean, _ in results[key]])
    best_std = [std for mean, std in results[key] if mean == best_mean][0]

    components = []

    zero_score = zero_scores[key]
    one_score = one_scores[key]


    for mean, std in results[key]:
        # perform a t-test
        result = scipy.stats.ttest_ind_from_stats(best_mean, best_std * np.sqrt(n_trials - 1), n_trials, mean,
                                                  std * np.sqrt(n_trials - 1), n_trials, equal_var=False)

        norm_mean = (mean - zero_score) / (one_score - zero_score)
        # if significantly worse
        if 1 - result.pvalue / 2 < 0.95 or mean >= best_mean:
            components.append("$\\bf %.2f\quad(%.2f)$" % (mean, norm_mean))#, std))
        else:
            components.append("$%.2f\quad(%.2f)$" % (mean, norm_mean))#, std))


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
