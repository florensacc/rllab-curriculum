from __future__ import print_function
import flask  # import Flask, render_template, send_from_directory
from rllab.misc.ext import flatten
from rllab.viskit import core
from rllab.misc import ext
import sys
import argparse
import json
import numpy as np
# import threading, webbrowser
import plotly.offline as po
import plotly.graph_objs as go
import pdb


USE_MEDIAN = False

app = flask.Flask(__name__, static_url_path='/static')

exps_data = None
plottable_keys = None
distinct_params = None


@app.route('/js/<path:path>')
def send_js(path):
    return flask.send_from_directory('js', path)


@app.route('/css/<path:path>')
def send_css(path):
    return flask.send_from_directory('css', path)


def make_plot(plot_list):
    data = []
    for idx, plt in enumerate(plot_list):
        color = core.color_defaults[idx % len(core.color_defaults)]
        if USE_MEDIAN:
            x = range(len(plt.percentile50))
            y = list(plt.percentile50)
            y_upper = list(plt.percentile75)
            y_lower = list(plt.percentile25)
        else:
            x = range(len(plt.means))
            y = list(plt.means)
            y_upper = list(plt.means + plt.stds)
            y_lower = list(plt.means - plt.stds)

        data.append(go.Scatter(
            x=x + x[::-1],
            y=y_upper + y_lower[::-1],
            fill='tozerox',
            fillcolor=core.hex_to_rgb(color, 0.2),
            line=go.Line(color='transparent'),
            showlegend=False,
            legendgroup=plt.legend,
            hoverinfo='none'
        ))
        data.append(go.Scatter(
            x=x,
            y=y,
            name=plt.legend,
            legendgroup=plt.legend,
            line=dict(color=core.hex_to_rgb(color)),
        ))

    layout = go.Layout(
        legend=dict(
            xanchor="auto",
            yanchor="bottom",
        )
    )
    fig = go.Figure(data=data, layout=layout)
    return po.plot(fig, output_type='div', include_plotlyjs=False)


def get_plot_instruction(plot_key, split_key=None, group_key=None, filters={}):
    print(plot_key, split_key, group_key, filters)
    selector = core.Selector(exps_data)
    for k, v in filters.iteritems():
        selector = selector.where(k, str(v))
    # print selector._filters
    if split_key is not None:
        vs = [vs for k, vs in distinct_params if k == split_key][0]
        split_selectors = [selector.where(split_key, v) for v in vs]
        split_legends = map(str, vs)
    else:
        split_selectors = [selector]
        split_legends = ["Plot"]
    plots = []
    for split_selector, split_legend in zip(split_selectors, split_legends):
        if group_key:
            vs = [vs for k, vs in distinct_params if k == group_key][0]
            group_selectors = [split_selector.where(group_key, v) for v in vs]
            group_legends = map(str, vs)
        else:
            group_selectors = [split_selector]
            group_legends = [split_legend]
        to_plot = []
        for group_selector, group_legend in zip(group_selectors, group_legends):
            filtered_data = group_selector.extract()
            if len(filtered_data) > 0:

                # Group by seed and sort.
                # -----------------------
                filtered_params = core.extract_distinct_params(
                    filtered_data, l=0)
                filtered_params2 = [p[1] for p in filtered_params]
                filtered_params_k = [p[0] for p in filtered_params]
                import itertools
                product_space = itertools.product(
                    *filtered_params2
                )
                data_best_regret = None
                best_regret = -np.inf
                for params in product_space:
                    selector = core.Selector(exps_data)
                    for k, v in zip(filtered_params_k, params):
                        selector = selector.where(k, str(v))
                    data = selector.extract()
                    if len(data) > 0:
                        progresses = [
                            exp.progress.get(plot_key, np.array([np.nan])) for exp in data]
                        sizes = map(len, progresses)
                        max_size = max(sizes)
                        progresses = [
                            np.concatenate([ps, np.ones(max_size - len(ps)) * np.nan]) for ps in progresses]

                        if USE_MEDIAN:
                            medians = np.nanmedian(progresses, axis=0)
                            regret = np.median(medians)
                        else:
                            means = np.nanmean(progresses, axis=0)
                            regret = np.mean(means)
                        if regret > best_regret:
                            best_regret = regret
                            data_best_regret = data
                        distinct_params_k = [p[0] for p in distinct_params]
                        distinct_params_v = [
                            v for k, v in zip(filtered_params_k, params) if k in distinct_params_k]
                        distinct_params_kv = [
                            (k, v) for k, v in zip(distinct_params_k, distinct_params_v)]
                        distinct_params_kv_string = str(
                            distinct_params_kv).replace('), ', ')\t')
                        print(
                            '{}\t{}\t{}'.format(regret, len(progresses), distinct_params_kv_string))

                print(group_selector._filters)
                print('best regret: {}'.format(best_regret))
                # -----------------------
                if best_regret != -np.inf:
                    progresses = [
                        exp.progress.get(plot_key, np.array([np.nan])) for exp in data_best_regret]
                    sizes = map(len, progresses)
                    # more intelligent:
                    max_size = max(sizes)
                    progresses = [
                        np.concatenate([ps, np.ones(max_size - len(ps)) * np.nan]) for ps in progresses]
                    legend = '{} ({:.1f})'.format(
                        group_legend, best_regret)
                    if USE_MEDIAN:
                        percentile25 = np.nanpercentile(
                            progresses, q=25, axis=0)
                        percentile50 = np.nanpercentile(
                            progresses, q=50, axis=0)
                        percentile75 = np.nanpercentile(
                            progresses, q=75, axis=0)
                        to_plot.append(
                            ext.AttrDict(percentile25=percentile25, percentile50=percentile50,
                                         percentile75=percentile75, legend=legend))
                    else:
                        means = np.nanmean(progresses, axis=0)
                        stds = np.nanstd(progresses, axis=0)
                        to_plot.append(
                            ext.AttrDict(means=means, stds=stds, legend=legend))

        if len(to_plot) > 0:
            plots.append("<div>%s: %s</div>" % (split_key, split_legend))
            plots.append(make_plot(to_plot))
    return "\n".join(plots)


@app.route("/plot_div")
def plot_div():
    args = flask.request.args
    plot_key = args.get("plot_key")
    split_key = args.get("split_key", "")
    group_key = args.get("group_key", "")
    filters_json = args.get("filters", "{}")
    filters = json.loads(filters_json)
    if len(split_key) == 0:
        split_key = None
    if len(group_key) == 0:
        group_key = None
    # group_key = distinct_params[0][0]
    # print split_key
    # exp_filter = distinct_params[0]
    plot_div = get_plot_instruction(plot_key=plot_key, split_key=split_key,
                                    group_key=group_key, filters=filters)
    # print plot_div
    return plot_div


@app.route("/")
def index():
    # exp_folder_path = "data/s3/experiments/ppo-atari-3"
    # _load_data(exp_folder_path)
    # exp_json = json.dumps(exp_data)
    plot_key = "AverageReturn"
    group_key = distinct_params[0][0]
    plot_div = get_plot_instruction(
        plot_key=plot_key, split_key=None, group_key=group_key)
    return flask.render_template(
        "main.html",
        plot_div=plot_div,
        plot_key=plot_key,
        plottable_keys=plottable_keys,
        distinct_param_keys=[str(k) for k, v in distinct_params],
        distinct_params=dict([(str(k), map(str, v))
                              for k, v in distinct_params]),
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args(sys.argv[1:])
    print("Importing data from {path}...".format(path=args.data_path))
    exps_data = core.load_exps_data(args.data_path)
    plottable_keys = list(
        set(flatten(exp.progress.keys() for exp in exps_data)))

    distinct_params = core.extract_distinct_params(exps_data)
    port = 5000
    url = "http://127.0.0.1:{0}".format(port)
    print("Done! View http://localhost:5000 in your browser")
    app.run(debug=args.debug)
