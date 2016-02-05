import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
import csv
from glob import glob

## lr_perf = []
def plot_experiments(
    name_or_patterns,
    legend=False,
    post_processing=None,
    key='AverageReturn',
    get_color=None,
    plot=True,
    aggregate=False,
    row_reader=None,
    matcher=None,
):
    if not isinstance(name_or_patterns, (list, tuple)):
        name_or_patterns = [name_or_patterns]
    data_folder = osp.abspath('data')
#     print data_folder
    files = []
    for name_or_pattern in name_or_patterns:
        matched_files = glob(osp.join(data_folder, name_or_pattern))
        files += matched_files
    files = sorted(files)
#     print 'plotting the following experiments:'
    plots = []
    legends = []
    all_returns = {}
    for f in files:
        if matcher is not None:
            if not matcher(f):
                continue
        try:
#             print f
            exp_name = osp.basename(f)
            returns = []
            isnan = False
            skipped = False
            with open(osp.join(f, 'progress.csv'), 'rb') as csvfile:
                reader = csv.DictReader(csvfile)
                cnt = 0
                for row in reader:
                    cnt += 1
                    # if cnt <= 3:
                    #     continue
                    if row_reader is not None:
                        maybe = row_reader(row)
                        if maybe is not None:
                            returns.append(maybe)
                    else:
                        if key not in row:
                            skipped = True
                            break
                        if row[key]:
                            returns.append(float(row[key]))
            if skipped:
                continue
            if len(returns) == 0:
                continue
            returns = np.array(returns)
            if post_processing:
                returns = post_processing(returns)
            all_returns[exp_name] = returns
            if plot and not aggregate:
                plot = plt.plot(returns)[0]
                if get_color:
                    plot.set_color(get_color(exp_name))
                plots.append(plot)
                legends.append(exp_name)
        except KeyboardInterrupt:
            raise
        except Exception,e:
            print "error processing: ", f
            print str(e)
    if legend:
        plt.legend(plots, legends, bbox_to_anchor=(1, 0.2))
    if aggregate:
        all_s = all_returns.values()
        max_len = map(len, all_s)
        filter_s = [s for s in all_s if len(s) == max_len]
        raws = zip(all_s)
        all_lens = np.array(map(len, raws))
        assert np.alltrue(all_lens == max(all_lens))
        means = np.mean(raws, axis=0).flatten()
        stds = np.std(raws, axis=0).flatten()
        plot = plt.plot(means)[0]
        if get_color:
            color = get_color(name_or_patterns)
            plot.set_color(color)
        else:
            color = plot.get_color()
#         import pdb; pdb.set_trace()
        plt.fill_between(plot.get_xdata(), means-2*stds, means+2*stds,
            alpha=0.3, facecolor=color,
            linewidth=0)
    return all_returns

import statsmodels.api as sm
import numpy as np
def smooth_plot(ys):
    xs = np.arange(len(ys))
    return sm.nonparametric.lowess(ys, xs, frac=0.2)[:, 1]

import matplotlib.patches as mpatches

def plot_std(means, stds):
    plot = plt.plot(means)[0]
    color = plot.get_color()
    plt.fill_between(plot.get_xdata(), means-2*stds, means+2*stds,
        alpha=0.3, facecolor=color,
        linewidth=0)

def batch_plot(*args, **kwargs):
    patches = []
    colors = "bgrcmykw"
    key = kwargs.get("key", 'AverageReturn')
    for i, arg in enumerate(args):
        try:
            fn, name = arg
        except:
            fn, name = arg, arg
        color = colors[i]
        print color
        patch = mpatches.Patch(color=color, label=name)
        patches.append(patch)
        plot_experiments(
            fn,
            key=key,
            get_color=lambda _: color,
            aggregate=kwargs.get("aggregate", True),
            row_reader=kwargs.get("row_reader"),
            matcher=kwargs.get("matcher"),
        )

    plt.legend(handles=patches, bbox_to_anchor=(0., -0.2))

import json
def get_params(fn):
    data_folder = osp.abspath('data')
    fn = osp.join(data_folder, fn)
    with open(osp.join(fn, 'params.json'), 'rb') as jsonfile:
        param_h = json.load(jsonfile)
        return param_h

def mk_matcher(*args, **h):
    if len(args) != 0:
        assert len(args) == 1
        h = args[0]
    def match(h1, h2):
        if h2 is None:
            return False
        for k, v in h1.items():
            v2 = h2.get(k)
            if isinstance(v, dict):
                if not match(v, v2):
                    return False
            else:
                if v != v2:
                    return False
        return True
    def matcher(fn):
        return match(h, get_params(fn))
    return matcher

# Dict FileName Series -> (Series -> Number)? -> [(param, mean_series, std_series, [series]) sorted by key mean_series]
# different seeds will be grouped into same param
import json
from collections import defaultdict
def group_by_params(rets, key=lambda x: -x[-1]):
    def conv(h):
        h = dict(h)
        if "seed" in h:
            del h["seed"]
        if "exp_name" in h:
            del h["exp_name"]
        return json.dumps(h, sort_keys=True)
    seen = defaultdict(list)
    for filename, series in rets.items():
        params = get_params(filename)
        seen[conv(params)].append(series)
    outs = []
    for param_str, lst_series in seen.items():
        params = json.loads(param_str)
        max_len = max([len(s) for s in lst_series])
        eligible_lst_series = [s for s in lst_series if len(s) == max_len]
        if len(lst_series) != len(eligible_lst_series):
            print "warning: some series have imcomplete data, ignoring"
        mean_series = np.mean(eligible_lst_series, axis=0).flatten()
        std_series = np.std(eligible_lst_series, axis=0).flatten()
        outs.append(
            (params, mean_series, std_series, lst_series)
        )
    return sorted(outs, key=lambda o: key(o[1]))

# [Hash] -> (Common Hash, [Unique Hash]) in flatten format
def factorize_params(lst_hash):
    from __builtin__ import sum
    def flatten(h):
        return sum([
            [("%s_%s" % (k, kn), vn) for kn, vn in flatten(v)]
                if isinstance(v, dict) else [(k, v)]
                for k, v in h.items()
        ], [])
    def st(t): # serialize tuple
        return tuple(map(str, t))
    flatten_lst_hash = map(flatten, lst_hash)
    str_all_items_set = set(map(st, sum(flatten_lst_hash, [])))
    s_pairs = [map(st, pairs) for pairs in flatten_lst_hash]
    str_common_items_set = str_all_items_set.intersection(*s_pairs)
    return (str_common_items_set, [
            sorted(list(set(sp).difference(str_common_items_set))) for sp in s_pairs
        ])


def comp(pattern, **kwargs):
    kwargs["plot"] = kwargs.get("plot", False)
    mstd = kwargs.pop("mstd", False)
    rets=plot_experiments(
        pattern,
        **kwargs
    )
    plt.figure()
    groups=group_by_params(rets)
    if mstd:
        # mean - std
        groups = sorted(groups, key=lambda group: -np.mean(group[1] - group[2]))
    common, factors = factorize_params(map(lambda x: x[0], groups))
    for group in groups:
        plot_std(group[1], group[2])
    plt.legend(map(str, factors), loc='upper center', bbox_to_anchor=(0.5, -0.05))
    return (rets, groups, common, factors)

