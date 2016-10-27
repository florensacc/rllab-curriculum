from io import StringIO

gittins_results = {
    2: [
        (10, 5.99),
        (50, 31.811),
        (100, 64.21),
        (500, 318.86),
    ],
    5: [
        (10, 6.6068),
        (50, 37.55),
        (100, 78.744),
        (500, 401.89),
    ],
    10: [
        (10, 6.679),
        (50, 39.92),
        (100, 83.632),
        (500, 439.79),
    ],
    50: [
        (10, 6.6274),
        (50, 40.095),
        (100, 84.978),
        (500, 458.71),
    ]
}

rnn_results = {
    # 2: [
    #     (10, 6.0018),
    #     (50, 31.516),
    #     (100, 64.172),
    #     (500, 325.29),
    # ],
    5: [
        (10, 6.616),
        (100, 74.404),
        (500, 412.622),
    ],
    10: [
        (10, 6.70592),
        (100, 81.2792),
        (500, 432.058),
    ],
    50: [
        (10, 6.5828),
        (100, 73.386),
        (500, 438.602),
    ]
}




TEMPLATE = r"""
\begin{table}[th]
\caption{MAB Results}
\label{mab-table}
\begin{center}
\begin{tabular}{lll}
\multicolumn{1}{c}{\bf Setup}  &\multicolumn{1}{c}{\bf Gittins Index} & \multicolumn{1}{c}{\bf RNN Policy}
\\ \hline \\
%s
\end{tabular}
\end{center}
\end{table}
"""

selected_settings = [
    # (2, 10),
    # (2, 100),
    # (2, 500),
    (5, 10),
    (5, 100),
    (5, 500),
    (10, 10),
    (10, 100),
    (10, 500),
    (50, 10),
    (50, 100),
    (50, 500),
]


buf = StringIO()
for K, T in selected_settings:
    buf.write(
        "$K={0}, T={1}$".format(K, T)
    )
    buf.write(" & ")
    buf.write("$%.2f$" % dict(gittins_results[K])[T])
    buf.write(" & ")
    buf.write("$%.2f$" % dict(rnn_results[K])[T])
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

