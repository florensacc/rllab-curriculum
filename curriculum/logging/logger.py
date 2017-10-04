import datetime
import os

import rllab.misc.logger
from curriculum.utils import AttrDict


class ExperimentLogger(object):
    def __init__(self, base_dir, itr=None, snapshot_mode='all', text_file='debug.log',
                 tabular_file='progress.csv', snapshot_gap=1, hold_outter_log=False):
        """
        :param base_dir:
        :param itr:
        :param snapshot_mode:
        :param text_file:
        :param tabular_file:
        :param snapshot_gap:
        :param outter_tabular:
        :param outter_text:
        """

        if itr is not None:
            self.log_dir = os.path.join(base_dir, 'itr_{}'.format(itr))
        else:
            self.log_dir = base_dir
        self.text_file = os.path.join(self.log_dir, text_file)
        self.tabular_file = os.path.join(self.log_dir, tabular_file)
        self.snapshot_mode = snapshot_mode
        self.snapshot_gap = snapshot_gap
        self.hold_outter_log = hold_outter_log

    def __enter__(self):
        self.prev_snapshot_dir = rllab.misc.logger.get_snapshot_dir()
        self.prev_mode = rllab.misc.logger.get_snapshot_mode()
        # if you want to avoid dumping the data to the outer csv or log file.
        if self.hold_outter_log:
            self.prev_tabular_file = rllab.misc.logger._tabular_outputs[0]
            self.prev_text_file = rllab.misc.logger._text_outputs[0]
            rllab.misc.logger.hold_tabular_output(self.prev_tabular_file)
            rllab.misc.logger.remove_text_output(self.prev_text_file)
        rllab.misc.logger.add_text_output(self.text_file)
        rllab.misc.logger.add_tabular_output(self.tabular_file)
        rllab.misc.logger.set_snapshot_dir(self.log_dir)
        rllab.misc.logger.set_snapshot_mode(self.snapshot_mode)
        rllab.misc.logger.set_snapshot_gap(self.snapshot_gap)
        return self

    def __exit__(self, type, value, traceback):
        if self.hold_outter_log:
            rllab.misc.logger.add_tabular_output(self.prev_tabular_file)
            rllab.misc.logger.add_text_output(self.prev_text_file)
        rllab.misc.logger.set_snapshot_mode(self.prev_mode)
        rllab.misc.logger.set_snapshot_dir(self.prev_snapshot_dir)
        rllab.misc.logger.remove_tabular_output(self.tabular_file)
        rllab.misc.logger.remove_text_output(self.text_file)


def format_experiment_log_path(script, experiment_type):
    experiment_date_host = datetime.datetime.today().strftime(
        "%Y-%m-%d_%H-%M-%S_{}".format(os.uname()[1])
    )

    experiment_data_dir = os.path.join(
        os.path.abspath(os.path.dirname(script)),
        "experiment_data/{}/{}".format(experiment_type, experiment_date_host)
    )

    plot_dir = os.path.join(experiment_data_dir, 'plot')
    log_dir = os.path.join(experiment_data_dir, 'log')
    report_dir = os.path.join(
        os.path.abspath(os.path.dirname(script)),
        'reports'
    )
    report_file = os.path.join(
        os.path.abspath(os.path.dirname(script)),
        datetime.datetime.today().strftime(
            "reports/{}_{}.html".format(experiment_date_host, experiment_type)
        )
    )

    return AttrDict(
        experiment_date_host=experiment_date_host,
        experiment_data_dir=experiment_data_dir,
        plot_dir=plot_dir,
        log_dir=log_dir,
        report_dir=report_dir,
        report_file=report_file,
    )


def make_log_dirs(log_path_config):
    os.makedirs(log_path_config.experiment_data_dir)
    os.makedirs(log_path_config.plot_dir)
    os.makedirs(log_path_config.log_dir)
    os.makedirs(log_path_config.report_dir, exist_ok=True)
