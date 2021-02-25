# -*- coding: utf-8 -*-
"""Parser for phys2denoise."""


import argparse

from phys2denoise import __version__
from phys2denoise.metrics.cardiac import crf
from phys2denoise.metrics.chest_belt import rpv, rv, rvt, rrf, env


class MetricsArgDict(argparse.Action):
    """
    Custom Argparse Action to create a dictionary with the metrics' arguments in parser's output.

    """
    def __call__(self, parser, namespace, values, option_strings):
        if not hasattr(namespace, "metrics_arg"):
            setattr(namespace, "metrics_arg", dict())
            Keys = ["sample_rate", "peaks", "throughs", "oversampling", "time_length", "onset",
                    "tr", "window", "lags", "nscans", "nharm"]
            Vals = ["None", "None", "None", "50", "None", "0", "None", "6", "None", "1", "None"]
            for k, v in zip(Keys, Vals):
                getattr(namespace, "metrics_arg")[k] = v
        getattr(namespace, "metrics_arg")[self.dest] = values


def _get_parser():
    """
    Parse command line inputs for this function.

    Returns
    -------
    parser.parse_args() : argparse dict

    Notes
    -----
    Default values must be updated in __call__ method from MetricsArgDict class.
    # Argument parser follow template provided by RalphyZ.
    # https://stackoverflow.com/a/43456577
    """

    parser = argparse.ArgumentParser()
    optional = parser._action_groups.pop()
    required = parser.add_argument_group("Required Argument")
    metric = parser.add_argument_group("Metrics")
    metric_arg = parser.add_argument_group("Metrics Arguments")
    required.add_argument("-in", "--input-file",
                          dest="filename",
                          type=str,
                          help="Full path and name of the file containing "
                               "physiological data, with or without extension.",
                          required=True)
    metric.add_argument("-crf", "--cardiac-response-function",
                        dest="metrics",
                        action="append_const",
                        const=crf,
                        help="Cardiac response function. Needs the following "
                             "inputs:sample-rate, oversampling, time-length, "
                             "onset and tr.",
                        default=[])
    metric.add_argument("-rpv", "--respiratory-pattern-variability",
                        dest="metrics",
                        action="append_const",
                        const=rpv,
                        help="Respiratory pattern variability. Needs the following "
                             "input: window.",
                        default=[])
    metric.add_argument("-env", "--envelope",
                        dest="metrics",
                        action="append_const",
                        const=env,
                        help="Respiratory pattern variability calculated across a sliding "
                             "window. Needs the following inputs: sample-rate, window and lags.",
                        default=[])
    metric.add_argument("-rv", "--respiratory-variance",
                        dest="metrics",
                        action="append_const",
                        const=rv,
                        help="Respiratory variance. Needs the following inputs: "
                             "sample-rate, window and lags.",
                        default=[])
    """
    metric.add_argument("-rvt", "--respiratory-volume-per-time",
                        dest="metrics",
                        action="append_const",
                        const="rvt",
                        help="Respiratory volume-per-time. Needs the following inputs: "
                             "sample-rate, window, lags, peaks and troughs.",
                        default=[])
     """
    metric.add_argument("-rrf", "--respiratory-response-function",
                        dest="metrics",
                        action="append_const",
                        const=rrf,
                        help="Respiratory response function. Needs the following inputs: "
                             "sample-rate, oversampling, time-length, onset and tr.",
                        default=[])
    metric.add_argument("-rcard", "--retroicor-card",
                        dest="metrics",
                        action="append_const",
                        const="r_card",
                        help="Computes regressors for cardiac signal. Needs the following "
                             "inputs: tr, nscans and n_harm.",
                        default=[])
    metric.add_argument("-rresp", "--retroicor-resp",
                        dest="metrics",
                        action="append_const",
                        const="r_resp",
                        help="Computes regressors for respiratory signal. Needs the following  "
                             "inputs: tr, nscans and n_harm.",
                        default=[])
    optional.add_argument("-outdir", "--output-dir",
                          dest="outdir",
                          type=str,
                          help="Folder where output should be placed. "
                               "Default is current folder.",
                          default=".")
    metric_arg.add_argument("-sr", "--sample-rate",
                            dest="sample_rate",
                            type=float,
                            action=MetricsArgDict,
                            help="Sampling rate of the physiological data in Hz.",
                            default=argparse.SUPPRESS)
    metric_arg.add_argument("-pk", "--peaks",
                            dest="peaks",
                            type=str,
                            action=MetricsArgDict,
                            help="Full path and filename of the list with the indexed peaks' "
                                 "positions of the physiological data.",
                            default=argparse.SUPPRESS)
    metric_arg.add_argument("-tg", "--troughs",
                            dest="troughs",
                            type=str,
                            action=MetricsArgDict,
                            help="Full path and filename of the list with the indexed troughs' "
                                 "positions of the physiological data.",
                            default=argparse.SUPPRESS)
    metric_arg.add_argument("-os", "--oversampling",
                            dest="oversampling",
                            type=int,
                            action=MetricsArgDict,
                            help="Temporal oversampling factor. "
                                 "Default is 50.",
                            default=argparse.SUPPRESS)
    metric_arg.add_argument("-tl", "--time-length",
                            dest="time_length",
                            type=int,
                            action=MetricsArgDict,
                            help="RRF or CRF Kernel length in seconds.",
                            default=argparse.SUPPRESS)
    metric_arg.add_argument("-onset", "--onset",
                            dest="onset",
                            type=float,
                            action=MetricsArgDict,
                            help="Onset of the response in seconds. "
                                 "Default is 0.",
                            default=argparse.SUPPRESS)
    metric_arg.add_argument("-tr", "--tr",
                            dest="tr",
                            type=float,
                            action=MetricsArgDict,
                            help="TR of sequence in seconds.",
                            default=argparse.SUPPRESS)
    metric_arg.add_argument("-win", "--window",
                            dest="window",
                            type=int,
                            action=MetricsArgDict,
                            help="Size of the sliding window in seconds. "
                                 "Default is 6 seconds.",
                            default=argparse.SUPPRESS)
    metric_arg.add_argument("-lags", "--lags",
                            dest="lags",
                            nargs="*",
                            type=int,
                            action=MetricsArgDict,
                            help="List of lags to apply to the RV estimate "
                                 "in seconds.",
                            default=argparse.SUPPRESS)
    metric_arg.add_argument("-nscans", "--number-scans",
                            dest="nscans",
                            type=int,
                            action=MetricsArgDict,
                            help="Number of scans. Default is 1.",
                            default=argparse.SUPPRESS)
    metric_arg.add_argument("-nharm", "--number-harmonics",
                            dest="n_harm",
                            type=int,
                            action=MetricsArgDict,
                            help="Number of harmonics.",
                            default=argparse.SUPPRESS)
    optional.add_argument("-debug", "--debug",
                          dest="debug",
                          action="store_true",
                          help="Only print debugging info to log file. Default is False.",
                          default=False)
    optional.add_argument("-quiet", "--quiet",
                          dest="quiet",
                          action="store_true",
                          help="Only print warnings to log file. Default is False.",
                          default=False)
    optional.add_argument("-v", "--version", action="version",
                          version=("%(prog)s " + __version__))

    parser._action_groups.append(optional)

    return parser


if __name__ == "__main__":
    raise RuntimeError("phys2denoise/cli/run.py should not be run directly;\n"
                       "Please `pip install` phys2denoise and use the "
                       "`phys2denoise` command")
