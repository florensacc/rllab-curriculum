from string import Template

results = [{'max_samples': '1000000', 'return': 67.949884454662111, 'name': 'AntEnv', 'n_workers': '1',
            'soft_target_tau': '0.0001', 'scale_reward': '0.1', 'batch_size': '16', 'qf_learning_rate': '0.0001',
            'policy_learning_rate': '0.0001', 'use_replay_pool': 'False'},
           {'max_samples': '1000000', 'return': 298.87919211803199, 'name': 'AntEnv', 'n_workers': '1',
            'soft_target_tau': '0.001', 'scale_reward': '0.01', 'batch_size': '32', 'qf_learning_rate': '0.0001',
            'policy_learning_rate': '1e-05', 'use_replay_pool': 'True'},
           {'max_samples': '4000000', 'return': 87.399389734444981, 'name': 'AntEnv', 'n_workers': '4',
            'soft_target_tau': '0.0001', 'scale_reward': '0.1', 'batch_size': '16', 'qf_learning_rate': '0.0001',
            'policy_learning_rate': '0.0001', 'use_replay_pool': 'False'},
           {'max_samples': '4000000', 'return': 504.04495125935887, 'name': 'AntEnv', 'n_workers': '4',
            'soft_target_tau': '0.0001', 'scale_reward': '0.001', 'batch_size': '32', 'qf_learning_rate': '1e-05',
            'policy_learning_rate': '1e-05', 'use_replay_pool': 'True'},
           {'max_samples': '8000000', 'return': 192.4657953830781, 'name': 'AntEnv', 'n_workers': '8',
            'soft_target_tau': '0.001', 'scale_reward': '0.01', 'batch_size': '16', 'qf_learning_rate': '0.0001',
            'policy_learning_rate': '0.0001', 'use_replay_pool': 'False'},
           {'max_samples': '8000000', 'return': 985.85828474446475, 'name': 'AntEnv', 'n_workers': '8',
            'soft_target_tau': '0.0001', 'scale_reward': '0.01', 'batch_size': '32', 'qf_learning_rate': '1e-05',
            'policy_learning_rate': '1e-06', 'use_replay_pool': 'True'},
           {'max_samples': '16000000', 'return': 307.64761696369715, 'name': 'AntEnv', 'n_workers': '16',
            'soft_target_tau': '0.001', 'scale_reward': '0.001', 'batch_size': '16', 'qf_learning_rate': '1e-05',
            'policy_learning_rate': '1e-05', 'use_replay_pool': 'False'},
           {'max_samples': '16000000', 'return': 1114.6542804062776, 'name': 'AntEnv', 'n_workers': '16',
            'soft_target_tau': '0.001', 'scale_reward': '0.01', 'batch_size': '32', 'qf_learning_rate': '1e-05',
            'policy_learning_rate': '1e-06', 'use_replay_pool': 'True'},
           {'max_samples': '1000000', 'return': 9.7146015693251826, 'name': 'SwimmerEnv', 'n_workers': '1',
            'soft_target_tau': '0.0001', 'scale_reward': '1', 'batch_size': '32', 'qf_learning_rate': '0.0001',
            'policy_learning_rate': '1e-05', 'use_replay_pool': 'False'},
           {'max_samples': '1000000', 'return': 87.665471506844625, 'name': 'SwimmerEnv', 'n_workers': '1',
            'soft_target_tau': '0.001', 'scale_reward': '1', 'batch_size': '32', 'qf_learning_rate': '0.001',
            'policy_learning_rate': '0.0001', 'use_replay_pool': 'True'},
           {'max_samples': '4000000', 'return': 9.8564892314774681, 'name': 'SwimmerEnv', 'n_workers': '4',
            'soft_target_tau': '0.0001', 'scale_reward': '1', 'batch_size': '16', 'qf_learning_rate': '1e-05',
            'policy_learning_rate': '1e-06', 'use_replay_pool': 'False'},
           {'max_samples': '4000000', 'return': 82.865131659118802, 'name': 'SwimmerEnv', 'n_workers': '4',
            'soft_target_tau': '0.001', 'scale_reward': '0.1', 'batch_size': '32', 'qf_learning_rate': '0.0001',
            'policy_learning_rate': '0.0001', 'use_replay_pool': 'True'},
           {'max_samples': '8000000', 'return': 8.2346409575220427, 'name': 'SwimmerEnv', 'n_workers': '8',
            'soft_target_tau': '0.0001', 'scale_reward': '1', 'batch_size': '32', 'qf_learning_rate': '1e-05',
            'policy_learning_rate': '1e-06', 'use_replay_pool': 'False'},
           {'max_samples': '8000000', 'return': 84.180923269292776, 'name': 'SwimmerEnv', 'n_workers': '8',
            'soft_target_tau': '0.0001', 'scale_reward': '0.1', 'batch_size': '32', 'qf_learning_rate': '0.0001',
            'policy_learning_rate': '0.0001', 'use_replay_pool': 'True'},
           {'max_samples': '16000000', 'return': 5.2072457841978821, 'name': 'SwimmerEnv', 'n_workers': '16',
            'soft_target_tau': '0.001', 'scale_reward': '0.01', 'batch_size': '4', 'qf_learning_rate': '1e-05',
            'policy_learning_rate': '1e-06', 'use_replay_pool': 'False'},
           {'max_samples': '16000000', 'return': 84.72197580675477, 'name': 'SwimmerEnv', 'n_workers': '16',
            'soft_target_tau': '0.001', 'scale_reward': '1', 'batch_size': '32', 'qf_learning_rate': '1e-05',
            'policy_learning_rate': '1e-05', 'use_replay_pool': 'True'},
           {'max_samples': '1000000', 'return': 1226.2023226587057, 'name': 'InvertedDoublePendulumEnv',
            'n_workers': '1',
            'soft_target_tau': '0.001', 'scale_reward': '0.1', 'batch_size': '4', 'qf_learning_rate': '0.0001',
            'policy_learning_rate': '1e-05', 'use_replay_pool': 'False'},
           {'max_samples': '1000000', 'return': 2589.9376308656701, 'name': 'InvertedDoublePendulumEnv',
            'n_workers': '1',
            'soft_target_tau': '0.001', 'scale_reward': '0.01', 'batch_size': '32', 'qf_learning_rate': '0.0001',
            'policy_learning_rate': '0.0001', 'use_replay_pool': 'True'},
           {'max_samples': '4000000', 'return': 2244.6759798196672, 'name': 'InvertedDoublePendulumEnv',
            'n_workers': '4',
            'soft_target_tau': '0.001', 'scale_reward': '0.01', 'batch_size': '4', 'qf_learning_rate': '0.001',
            'policy_learning_rate': '0.0001', 'use_replay_pool': 'False'},
           {'max_samples': '4000000', 'return': 2355.3963464410281, 'name': 'InvertedDoublePendulumEnv',
            'n_workers': '4',
            'soft_target_tau': '0.001', 'scale_reward': '0.1', 'batch_size': '32', 'qf_learning_rate': '1e-05',
            'policy_learning_rate': '1e-05', 'use_replay_pool': 'True'},
           {'max_samples': '8000000', 'return': 2230.7254180500872, 'name': 'InvertedDoublePendulumEnv',
            'n_workers': '8',
            'soft_target_tau': '0.001', 'scale_reward': '0.001', 'batch_size': '16', 'qf_learning_rate': '0.0001',
            'policy_learning_rate': '0.0001', 'use_replay_pool': 'False'},
           {'max_samples': '8000000', 'return': 1788.8412814876729, 'name': 'InvertedDoublePendulumEnv',
            'n_workers': '8',
            'soft_target_tau': '0.001', 'scale_reward': '0.001', 'batch_size': '32', 'qf_learning_rate': '1e-05',
            'policy_learning_rate': '1e-05', 'use_replay_pool': 'True'},
           {'max_samples': '16000000', 'return': 3861.6624029186146, 'name': 'InvertedDoublePendulumEnv',
            'n_workers': '16',
            'soft_target_tau': '0.001', 'scale_reward': '1', 'batch_size': '4', 'qf_learning_rate': '0.0001',
            'policy_learning_rate': '0.0001', 'use_replay_pool': 'False'},
           {'max_samples': '16000000', 'return': 2700.5897743162327, 'name': 'InvertedDoublePendulumEnv',
            'n_workers': '16',
            'soft_target_tau': '0.001', 'scale_reward': '0.001', 'batch_size': '32', 'qf_learning_rate': '0.0001',
            'policy_learning_rate': '1e-05', 'use_replay_pool': 'True'},
           {'max_samples': '1000000', 'return': 4.6400953565360021, 'name': 'CartpoleSwingupEnv', 'n_workers': '1',
            'soft_target_tau': '0.001', 'scale_reward': '1', 'batch_size': '4', 'qf_learning_rate': '0.001',
            'policy_learning_rate': '0.0001', 'use_replay_pool': 'False'},
           {'max_samples': '1000000', 'return': 197.5162130543109, 'name': 'CartpoleSwingupEnv', 'n_workers': '1',
            'soft_target_tau': '0.001', 'scale_reward': '1', 'batch_size': '32', 'qf_learning_rate': '0.001',
            'policy_learning_rate': '0.001', 'use_replay_pool': 'True'},
           {'max_samples': '4000000', 'return': -96.006881287928834, 'name': 'CartpoleSwingupEnv', 'n_workers': '4',
            'soft_target_tau': '0.0001', 'scale_reward': '0.1', 'batch_size': '4', 'qf_learning_rate': '1e-05',
            'policy_learning_rate': '1e-06', 'use_replay_pool': 'False'},
           {'max_samples': '4000000', 'return': 145.71790838995909, 'name': 'CartpoleSwingupEnv', 'n_workers': '4',
            'soft_target_tau': '0.001', 'scale_reward': '0.01', 'batch_size': '32', 'qf_learning_rate': '0.0001',
            'policy_learning_rate': '0.0001', 'use_replay_pool': 'True'},
           {'max_samples': '8000000', 'return': -74.553231385352078, 'name': 'CartpoleSwingupEnv', 'n_workers': '8',
            'soft_target_tau': '0.001', 'scale_reward': '1', 'batch_size': '32', 'qf_learning_rate': '0.0001',
            'policy_learning_rate': '0.0001', 'use_replay_pool': 'False'},
           {'max_samples': '8000000', 'return': 147.60239150489812, 'name': 'CartpoleSwingupEnv', 'n_workers': '8',
            'soft_target_tau': '0.0001', 'scale_reward': '0.1', 'batch_size': '32', 'qf_learning_rate': '1e-05',
            'policy_learning_rate': '1e-05', 'use_replay_pool': 'True'},
           {'max_samples': '16000000', 'return': -107.19407621268121, 'name': 'CartpoleSwingupEnv',
            'n_workers': '16',
            'soft_target_tau': '0.0001', 'scale_reward': '0.1', 'batch_size': '16', 'qf_learning_rate': '0.001',
            'policy_learning_rate': '0.001', 'use_replay_pool': 'False'},
           {'max_samples': '16000000', 'return': 72.83199437204054, 'name': 'CartpoleSwingupEnv', 'n_workers': '16',
            'soft_target_tau': '0.001', 'scale_reward': '0.001', 'batch_size': '32', 'qf_learning_rate': '1e-05',
            'policy_learning_rate': '1e-06', 'use_replay_pool': 'True'}]

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

for idx, (env, results) in enumerate(per_env.items()):
    print(("\\multicolumn{9}{c}{%s} \\\\" % env_names[env]))
    print("\\toprule")
    results = sorted(results, key=lambda x: (int(x["n_workers"]), x["use_replay_pool"]))
    keys = list(results[0].keys())
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
