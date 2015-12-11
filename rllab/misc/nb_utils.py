import os.path as osp
import numpy as np
import csv
import matplotlib.pyplot as plt
from glob import glob

def plot_experiments(name_or_patterns, legend=False, post_processing=None, key='AverageReturn'):
    if not isinstance(name_or_patterns, (list, tuple)):
        name_or_patterns = [name_or_patterns]
    data_folder = osp.abspath(osp.join(osp.dirname(__file__), '../../data'))
    files = []
    for name_or_pattern in name_or_patterns:
        matched_files = glob(osp.join(data_folder, name_or_pattern))
        files += matched_files
    files = sorted(files)
    print 'plotting the following experiments:'
    for f in files:
        print f
    plots = []
    legends = []
    for f in files:
        exp_name = osp.basename(f)
        returns = []
        with open(osp.join(f, 'progress.csv'), 'rb') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row[key]:
                    returns.append(float(row[key]))
        returns = np.array(returns)
        if post_processing:
            returns = post_processing(returns)
        plots.append(plt.plot(returns)[0])
        legends.append(exp_name)
    if legend:
        plt.legend(plots, legends)
