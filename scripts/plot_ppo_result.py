import numpy as np
import matplotlib.pyplot as plt
import re

def get_lines(file_name):
    with open(file_name) as f:
        return f.read().split('\n')

def extract_re(lines, pattern, group_id):
    matches = [re.search(pattern, line) for line in lines]
    return [match.group(group_id) for match in matches if match]

def extract_key(lines, key):
    return extract_re(lines, r'%s\s+([\d\.]+)' % key, 1)

def extract_stats(file_name):
    lines = get_lines(file_name)
    itrs = map(int, extract_key(lines, 'Iteration'))
    itr_start = len(itrs) - 1 - itrs[::-1].index(0)
    return dict(
        avg_returns=map(float, extract_key(lines, 'AvgReturn'))[itr_start:],
        ev=map(float, extract_key(lines, 'ExplainedVariance'))[itr_start:],
    )

def plot_and_compare(*exps):
    file_names = [file_name for file_name, _ in exps]
    exp_names = [exp_name for _, exp_name in exps]
    stats = map(extract_stats, file_names)
    avg_returns = [exp_stats["avg_returns"] for exp_stats in stats]
    min_length = min(map(len, avg_returns))

    plt.subplot(211)
    for exp_stats, exp_name in zip(stats, exp_names):
        plt.plot(exp_stats["avg_returns"][:min_length], label=exp_name)
    plt.legend(loc='lower right')

    plt.subplot(212)
    for exp_stats, exp_name in zip(stats, exp_names):
        plt.plot(exp_stats["ev"][:min_length], label=exp_name)
    plt.legend(loc='lower right')
    plt.show()

plot_and_compare(
    ('data/hopper_30k/log.txt', 'hopper_30k'),
    ('data/hopper_30k_log/log.txt', 'hopper_30k_log'),
    ('data/hopper_30k_sum/log.txt', 'hopper_30k_sum'),
    ('data/hopper_per_state_std_30k/log.txt', 'hopper_per_state_std_30k'),
    #('data/hopper_vf_100k/log.txt', 'hopper_vf_100k'),
    #('data/hopper_vf_100k_log/log.txt', 'hopper_vf_100k_log'),
    #('data/hopper_vf_100k_sum/log.txt', 'hopper_vf_100k_sum'),
    #('data/hopper_no_vf_100k/log.txt', 'hopper_no_vf_100k'),
    #('data/hopper_vf/log.txt', 'hopper_vf'),
    #('data/hopper_no_vf/log.txt', 'hopper_no_vf'),
)
