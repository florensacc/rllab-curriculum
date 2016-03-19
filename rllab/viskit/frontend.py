import flask #import Flask, render_template, send_from_directory
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
# exp_data = Nonecached_data = dict()
#
#
# def load_data(exp_folder_path):
#     if exp_folder_path not in cached_data:
#         cached_data[exp_folder_path] = core.load_exps_data(exp_folder_path)

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
            line=dict(color=core.hex_to_rgb(color))
        ))
    return po.plot(data, output_type='div')

def get_plot_instruction(plot_key, exp_filter):
    k, vs = exp_filter
    selector = core.Selector(exps_data)
    to_plot = []
    for v in vs:
        filtered_data = selector.where(k, v).extract()
        returns = [exp.progress[plot_key] for exp in filtered_data]
        sizes = map(len, returns)
        max_size = max(sizes)
        # for exp, retlen in zip(filtered_data, sizes):
        #     if retlen < max_size:
        #         log("Excluding {exp_name} since the trajectory is shorter: {thislen} vs. {maxlen}".format(
        #             exp_name=exp.params["exp_name"], thislen=retlen, maxlen=max_size))
        returns = [ret for ret in returns if len(ret) == max_size]
        mean_returns = np.mean(returns, axis=0)
        std_returns = np.std(returns, axis=0)
        # self._plot_sequence.append((''))
        to_plot.append(ext.AttrDict(means=mean_returns, stds=std_returns, legend=str(v)))
    return make_plot(to_plot)

@app.route("/plot_div")
def plot_div():
    plot_key = flask.request.args.get("plot_key")
    exp_filter = distinct_params[0]
    plot_div = get_plot_instruction(plot_key, exp_filter)
    return plot_div

@app.route("/")
def index():
    # exp_folder_path = "data/s3/experiments/ppo-atari-3"
    # _load_data(exp_folder_path)
    # exp_json = json.dumps(exp_data)
    plot_key = "AverageReturn"
    exp_filter = distinct_params[0]

    plot_div = get_plot_instruction(plot_key, exp_filter)
    return flask.render_template(
        "main.html",
        plot_div=plot_div,
        plot_key=plot_key,
        plottable_keys=plottable_keys
    )#exp_json=exp_json)
    # return "Hello World!"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args(sys.argv[1:])
    print("Importing data from {path}...".format(path=args.data_path))
    exps_data = core.load_exps_data(args.data_path)
    plottable_keys = exps_data[0].progress.keys()
    distinct_params = core.extract_distinct_params(exps_data)
    port = 5000
    url = "http://127.0.0.1:{0}".format(port)
    print("Done! View http://localhost:5000 in your browser")
    app.run(debug=args.debug)
