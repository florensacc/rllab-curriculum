from string import Template

results = [{u'max_samples': '1000000', 'return': 67.949884454662111, u'name': 'AntEnv', u'n_workers': '1',
            u'soft_target_tau': '0.0001', u'scale_reward': '0.1', u'batch_size': '16', u'qf_learning_rate': '0.0001',
            u'policy_learning_rate': '0.0001', u'use_replay_pool': 'False'},
           {u'max_samples': '1000000', 'return': 298.87919211803199, u'name': 'AntEnv', u'n_workers': '1',
            u'soft_target_tau': '0.001', u'scale_reward': '0.01', u'batch_size': '32', u'qf_learning_rate': '0.0001',
            u'policy_learning_rate': '1e-05', u'use_replay_pool': 'True'},
           {u'max_samples': '4000000', 'return': 87.399389734444981, u'name': 'AntEnv', u'n_workers': '4',
            u'soft_target_tau': '0.0001', u'scale_reward': '0.1', u'batch_size': '16', u'qf_learning_rate': '0.0001',
            u'policy_learning_rate': '0.0001', u'use_replay_pool': 'False'},
           {u'max_samples': '4000000', 'return': 504.04495125935887, u'name': 'AntEnv', u'n_workers': '4',
            u'soft_target_tau': '0.0001', u'scale_reward': '0.001', u'batch_size': '32', u'qf_learning_rate': '1e-05',
            u'policy_learning_rate': '1e-05', u'use_replay_pool': 'True'},
           {u'max_samples': '8000000', 'return': 192.4657953830781, u'name': 'AntEnv', u'n_workers': '8',
            u'soft_target_tau': '0.001', u'scale_reward': '0.01', u'batch_size': '16', u'qf_learning_rate': '0.0001',
            u'policy_learning_rate': '0.0001', u'use_replay_pool': 'False'},
           {u'max_samples': '8000000', 'return': 985.85828474446475, u'name': 'AntEnv', u'n_workers': '8',
            u'soft_target_tau': '0.0001', u'scale_reward': '0.01', u'batch_size': '32', u'qf_learning_rate': '1e-05',
            u'policy_learning_rate': '1e-06', u'use_replay_pool': 'True'},
           {u'max_samples': '16000000', 'return': 307.64761696369715, u'name': 'AntEnv', u'n_workers': '16',
            u'soft_target_tau': '0.001', u'scale_reward': '0.001', u'batch_size': '16', u'qf_learning_rate': '1e-05',
            u'policy_learning_rate': '1e-05', u'use_replay_pool': 'False'},
           {u'max_samples': '16000000', 'return': 1114.6542804062776, u'name': 'AntEnv', u'n_workers': '16',
            u'soft_target_tau': '0.001', u'scale_reward': '0.01', u'batch_size': '32', u'qf_learning_rate': '1e-05',
            u'policy_learning_rate': '1e-06', u'use_replay_pool': 'True'},
           {u'max_samples': '1000000', 'return': 9.7146015693251826, u'name': 'SwimmerEnv', u'n_workers': '1',
            u'soft_target_tau': '0.0001', u'scale_reward': '1', u'batch_size': '32', u'qf_learning_rate': '0.0001',
            u'policy_learning_rate': '1e-05', u'use_replay_pool': 'False'},
           {u'max_samples': '1000000', 'return': 87.665471506844625, u'name': 'SwimmerEnv', u'n_workers': '1',
            u'soft_target_tau': '0.001', u'scale_reward': '1', u'batch_size': '32', u'qf_learning_rate': '0.001',
            u'policy_learning_rate': '0.0001', u'use_replay_pool': 'True'},
           {u'max_samples': '4000000', 'return': 9.8564892314774681, u'name': 'SwimmerEnv', u'n_workers': '4',
            u'soft_target_tau': '0.0001', u'scale_reward': '1', u'batch_size': '16', u'qf_learning_rate': '1e-05',
            u'policy_learning_rate': '1e-06', u'use_replay_pool': 'False'},
           {u'max_samples': '4000000', 'return': 82.865131659118802, u'name': 'SwimmerEnv', u'n_workers': '4',
            u'soft_target_tau': '0.001', u'scale_reward': '0.1', u'batch_size': '32', u'qf_learning_rate': '0.0001',
            u'policy_learning_rate': '0.0001', u'use_replay_pool': 'True'},
           {u'max_samples': '8000000', 'return': 8.2346409575220427, u'name': 'SwimmerEnv', u'n_workers': '8',
            u'soft_target_tau': '0.0001', u'scale_reward': '1', u'batch_size': '32', u'qf_learning_rate': '1e-05',
            u'policy_learning_rate': '1e-06', u'use_replay_pool': 'False'},
           {u'max_samples': '8000000', 'return': 84.180923269292776, u'name': 'SwimmerEnv', u'n_workers': '8',
            u'soft_target_tau': '0.0001', u'scale_reward': '0.1', u'batch_size': '32', u'qf_learning_rate': '0.0001',
            u'policy_learning_rate': '0.0001', u'use_replay_pool': 'True'},
           {u'max_samples': '16000000', 'return': 5.2072457841978821, u'name': 'SwimmerEnv', u'n_workers': '16',
            u'soft_target_tau': '0.001', u'scale_reward': '0.01', u'batch_size': '4', u'qf_learning_rate': '1e-05',
            u'policy_learning_rate': '1e-06', u'use_replay_pool': 'False'},
           {u'max_samples': '16000000', 'return': 84.72197580675477, u'name': 'SwimmerEnv', u'n_workers': '16',
            u'soft_target_tau': '0.001', u'scale_reward': '1', u'batch_size': '32', u'qf_learning_rate': '1e-05',
            u'policy_learning_rate': '1e-05', u'use_replay_pool': 'True'},
           {u'max_samples': '1000000', 'return': 1226.2023226587057, u'name': 'InvertedDoublePendulumEnv',
            u'n_workers': '1',
            u'soft_target_tau': '0.001', u'scale_reward': '0.1', u'batch_size': '4', u'qf_learning_rate': '0.0001',
            u'policy_learning_rate': '1e-05', u'use_replay_pool': 'False'},
           {u'max_samples': '1000000', 'return': 2589.9376308656701, u'name': 'InvertedDoublePendulumEnv',
            u'n_workers': '1',
            u'soft_target_tau': '0.001', u'scale_reward': '0.01', u'batch_size': '32', u'qf_learning_rate': '0.0001',
            u'policy_learning_rate': '0.0001', u'use_replay_pool': 'True'},
           {u'max_samples': '4000000', 'return': 2244.6759798196672, u'name': 'InvertedDoublePendulumEnv',
            u'n_workers': '4',
            u'soft_target_tau': '0.001', u'scale_reward': '0.01', u'batch_size': '4', u'qf_learning_rate': '0.001',
            u'policy_learning_rate': '0.0001', u'use_replay_pool': 'False'},
           {u'max_samples': '4000000', 'return': 2355.3963464410281, u'name': 'InvertedDoublePendulumEnv',
            u'n_workers': '4',
            u'soft_target_tau': '0.001', u'scale_reward': '0.1', u'batch_size': '32', u'qf_learning_rate': '1e-05',
            u'policy_learning_rate': '1e-05', u'use_replay_pool': 'True'},
           {u'max_samples': '8000000', 'return': 2230.7254180500872, u'name': 'InvertedDoublePendulumEnv',
            u'n_workers': '8',
            u'soft_target_tau': '0.001', u'scale_reward': '0.001', u'batch_size': '16', u'qf_learning_rate': '0.0001',
            u'policy_learning_rate': '0.0001', u'use_replay_pool': 'False'},
           {u'max_samples': '8000000', 'return': 1788.8412814876729, u'name': 'InvertedDoublePendulumEnv',
            u'n_workers': '8',
            u'soft_target_tau': '0.001', u'scale_reward': '0.001', u'batch_size': '32', u'qf_learning_rate': '1e-05',
            u'policy_learning_rate': '1e-05', u'use_replay_pool': 'True'},
           {u'max_samples': '16000000', 'return': 3861.6624029186146, u'name': 'InvertedDoublePendulumEnv',
            u'n_workers': '16',
            u'soft_target_tau': '0.001', u'scale_reward': '1', u'batch_size': '4', u'qf_learning_rate': '0.0001',
            u'policy_learning_rate': '0.0001', u'use_replay_pool': 'False'},
           {u'max_samples': '16000000', 'return': 2700.5897743162327, u'name': 'InvertedDoublePendulumEnv',
            u'n_workers': '16',
            u'soft_target_tau': '0.001', u'scale_reward': '0.001', u'batch_size': '32', u'qf_learning_rate': '0.0001',
            u'policy_learning_rate': '1e-05', u'use_replay_pool': 'True'},
           {u'max_samples': '1000000', 'return': 4.6400953565360021, u'name': 'CartpoleSwingupEnv', u'n_workers': '1',
            u'soft_target_tau': '0.001', u'scale_reward': '1', u'batch_size': '4', u'qf_learning_rate': '0.001',
            u'policy_learning_rate': '0.0001', u'use_replay_pool': 'False'},
           {u'max_samples': '1000000', 'return': 197.5162130543109, u'name': 'CartpoleSwingupEnv', u'n_workers': '1',
            u'soft_target_tau': '0.001', u'scale_reward': '1', u'batch_size': '32', u'qf_learning_rate': '0.001',
            u'policy_learning_rate': '0.001', u'use_replay_pool': 'True'},
           {u'max_samples': '4000000', 'return': -96.006881287928834, u'name': 'CartpoleSwingupEnv', u'n_workers': '4',
            u'soft_target_tau': '0.0001', u'scale_reward': '0.1', u'batch_size': '4', u'qf_learning_rate': '1e-05',
            u'policy_learning_rate': '1e-06', u'use_replay_pool': 'False'},
           {u'max_samples': '4000000', 'return': 145.71790838995909, u'name': 'CartpoleSwingupEnv', u'n_workers': '4',
            u'soft_target_tau': '0.001', u'scale_reward': '0.01', u'batch_size': '32', u'qf_learning_rate': '0.0001',
            u'policy_learning_rate': '0.0001', u'use_replay_pool': 'True'},
           {u'max_samples': '8000000', 'return': -74.553231385352078, u'name': 'CartpoleSwingupEnv', u'n_workers': '8',
            u'soft_target_tau': '0.001', u'scale_reward': '1', u'batch_size': '32', u'qf_learning_rate': '0.0001',
            u'policy_learning_rate': '0.0001', u'use_replay_pool': 'False'},
           {u'max_samples': '8000000', 'return': 147.60239150489812, u'name': 'CartpoleSwingupEnv', u'n_workers': '8',
            u'soft_target_tau': '0.0001', u'scale_reward': '0.1', u'batch_size': '32', u'qf_learning_rate': '1e-05',
            u'policy_learning_rate': '1e-05', u'use_replay_pool': 'True'},
           {u'max_samples': '16000000', 'return': -107.19407621268121, u'name': 'CartpoleSwingupEnv',
            u'n_workers': '16',
            u'soft_target_tau': '0.0001', u'scale_reward': '0.1', u'batch_size': '16', u'qf_learning_rate': '0.001',
            u'policy_learning_rate': '0.001', u'use_replay_pool': 'False'},
           {u'max_samples': '16000000', 'return': 72.83199437204054, u'name': 'CartpoleSwingupEnv', u'n_workers': '16',
            u'soft_target_tau': '0.001', u'scale_reward': '0.001', u'batch_size': '32', u'qf_learning_rate': '1e-05',
            u'policy_learning_rate': '1e-06', u'use_replay_pool': 'True'}]

per_env = dict()
for result in results:
    env_name = result['name']
    if env_name not in per_env:
        per_env[env_name] = list()
    per_env[env_name].append(result)

print(r"""\begin{table}[h]
\caption{Best hyperparameters found for each task}
\label{best-hyperparams}
\centering
\begin{tabular}{lllllllll}
\toprule
""")


# $body
# \bottomrule
# \end{tabular}
#
# """)
#
# tmpl = Template(r"""\begin{table}[h]
# \caption{$title}
# \label{sample-table}
# \centering
# \begin{tabular}{lllllllll}
# \toprule
# $body
# \bottomrule
# \end{tabular}
# \end{table}""")


def format_val(key, val, params):
    if key == "use_replay_pool":
        return str(val)
    elif key == "max_samples":
        return str(val)
        # return "$" + str(params["n_workers"]) + "\\times 10^{"
    elif key == "return":
        return "%.2f" % float(val)
    return str(val)


ordered_keys = [
    "n_workers",
    "use_replay_pool",
    # "max_samples",
    "batch_size",
    "scale_reward",
    "qf_learning_rate",
    "policy_learning_rate",
    "soft_target_tau",
    "return",
]

key_names = dict(
    n_workers="Number of Workers",
    use_replay_pool="Use Replay Pool",
    batch_size="Batch Size",
    scale_reward="Reward Scaling",
    qf_learning_rate="Learning rate for $Q$",
    policy_learning_rate="Learning rate for policy",
    soft_target_tau="Soft target $\\tau$",
)
key_names["return"] = "Average Return"

env_names = dict(
    CartpoleSwingupEnv="Inverted Pendulum",
    InvertedDoublePendulumEnv="Double Inverted Pendulum",
    SwimmerEnv="Swimmer",
    AntEnv="Ant",
)

for idx, (env, results) in enumerate(per_env.iteritems()):
    print("\\multicolumn{9}{c}{%s} \\\\" % env_names[env])
    print("\\toprule")
    results = sorted(results, key=lambda x: (int(x["n_workers"]), x["use_replay_pool"]))
    keys = results[0].keys()
    body = ""
    for key in ordered_keys:
        body += key_names[key]
        for result in results:
            val = format_val(key, result[key], result)
            body += " & " + val
        body += "\\\\\n"
    print(body)
    if idx != len(per_env) - 1:
        print("\\bottomrule \\\\\\\\ \\toprule")
    else:
        print("\\bottomrule")

        # print(tmpl.substitute(title=env, body=body))
print("""\end{tabular}
\end{table}
""")
