import flask #import Flask, render_template, send_from_directory

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
        x = range(len(plt.means))
        y = list(plt.means)
        color = core.color_defaults[idx % len(core.color_defaults)]
        y_upper = list(plt.means + plt.stds)
        y_lower = list(plt.means - plt.stds)
        data.append(go.Scatter(
            x=x+x[::-1],
            y=y_upper+y_lower[::-1],
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


def get_plot_instruction(plot_key, split_key=None, group_key=None):
    print plot_key, split_key, group_key
    selector = core.Selector(exps_data)
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
            #import ipdb; ipdb.set_trace()
            filtered_data = group_selector.extract()
            if len(filtered_data) > 0:
                progresses = [exp.progress.get(plot_key, np.array([np.nan])) for exp in filtered_data]
                sizes = map(len, progresses)
                max_size = max(sizes)
                progresses = [ps for ps in progresses if len(ps) == max_size]
                means = np.mean(progresses, axis=0)
                stds = np.std(progresses, axis=0)
                to_plot.append(ext.AttrDict(means=means, stds=stds, legend=group_legend))
        if len(to_plot) > 0:
            plots.append(make_plot(to_plot))
    return "\n".join(plots)


@app.route("/plot_div")
def plot_div():
    args = flask.request.args
    plot_key = args.get("plot_key")
    split_key = args.get("split_key", "")
    group_key = args.get("group_key", "")
    if len(split_key) == 0:
        split_key = None
    if len(group_key) == 0:
        group_key = None
    # group_key = distinct_params[0][0]
    # print split_key
    # exp_filter = distinct_params[0]
    plot_div = get_plot_instruction(plot_key=plot_key, split_key=split_key, group_key=group_key)
    # print plot_div
    return plot_div


@app.route("/")
def index():
    # exp_folder_path = "data/s3/experiments/ppo-atari-3"
    # _load_data(exp_folder_path)
    # exp_json = json.dumps(exp_data)
    plot_key = "AverageReturn"
    group_key = distinct_params[0][0]
    plot_div = get_plot_instruction(plot_key=plot_key, split_key=None, group_key=group_key)
    return flask.render_template(
        "main.html",
        plot_div=plot_div,
        plot_key=plot_key,
        plottable_keys=plottable_keys,
        distinct_param_keys=[k for k, v in distinct_params]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args(sys.argv[1:])
    print("Importing data from {path}...".format(path=args.data_path))
    exps_data = core.load_exps_data(args.data_path)
    plottable_keys = list(set(flatten(exp.progress.keys() for exp in exps_data)))
    distinct_params = core.extract_distinct_params(exps_data)
    port = 5000
    url = "http://127.0.0.1:{0}".format(port)
    print("Done! View http://localhost:5000 in your browser")
    app.run(debug=args.debug)
