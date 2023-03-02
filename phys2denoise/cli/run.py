# -*- coding: utf-8 -*-
"""Parser for phys2denoise."""


import argparse

from phys2denoise import __version__
from phys2denoise.metrics.cardiac import crf
from phys2denoise.metrics.chest_belt import respiratory_pattern_variability, respiratory_variance, respiratory_variance_time, rrf, env


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
    metrics = parser.add_argument_group("Metrics")
    metric_arg = parser.add_argument_group("Metrics Arguments")
    # Required arguments
    required.add_argument("-in", "--input-file",
                          dest="filename",
                          type=str,
                          help="Full path and name of the file containing "
                               "physiological data, with or without extension.",
                          required=True)
    # Important optional arguments
    optional.add_argument("-outdir", "--output-dir",
                          dest="outdir",
                          type=str,
                          help="Folder where output should be placed. "
                               "Default is current folder.",
                          default=".")
    # Metric selection
    metrics.add_argument("-crf", "--cardiac-response-function",
                         dest="metrics",
                         action="append_const",
                         const=crf,
                         help="Cardiac response function. Requires the following "
                              "inputs:sample-rate, oversampling, time-length, "
                              "onset and tr.",
                         default=[])
    metrics.add_argument("-rpv", "--respiratory-pattern-variability",
                         dest="metrics",
                         action="append_const",
                         const=respiratory_pattern_variability,
                         help="Respiratory pattern variability. Requires the following "
                              "input: window.",
                         default=[])
    metrics.add_argument("-env", "--envelope",
                         dest="metrics",
                         action="append_const",
                         const=env,
                         help="Respiratory pattern variability calculated across a sliding "
                              "window. Requires the following inputs: sample-rate, window and lags.",
                         default=[])
    metrics.add_argument("-rv", "--respiratory-variance",
                         dest="metrics",
                         action="append_const",
                         const=respiratory_variance,
                         help="Respiratory variance. Requires the following inputs: "
                              "sample-rate, window and lags. If the input file "
                              "not a .phys file, it also requires peaks and troughs",
                         default=[])
    """
    metrics.add_argument("-rvt", "--respiratory-volume-per-time",
                         dest="metrics",
                         action="append_const",
                         const="respiratory_variance_time",
                         help="Respiratory volume-per-time. Requires the following inputs: "
                              "sample-rate, window, lags, peaks and troughs.",
                         default=[])
     """
    metrics.add_argument("-rrf", "--respiratory-response-function",
                         dest="metrics",
                         action="append_const",
                         const=rrf,
                         help="Respiratory response function. Requires the following inputs: "
                              "sample-rate, oversampling, time-length, onset and tr.",
                         default=[])
    metrics.add_argument("-rcard", "--retroicor-card",
                         dest="metrics",
                         action="append_const",
                         const="r_card",
                         help="Computes regressors for cardiac signal. Requires the following "
                              "inputs: tr, nscans and n_harm.",
                         default=[])
    metrics.add_argument("-rresp", "--retroicor-resp",
                         dest="metrics",
                         action="append_const",
                         const="r_resp",
                         help="Computes regressors for respiratory signal. Requires the following  "
                              "inputs: tr, nscans and n_harm.",
                         default=[])
    # Metric arguments
    metric_arg.add_argument("-sr", "--sample-rate",
                            dest="sample_rate",
                            type=float,
                            help="Sampling rate of the physiological data in Hz.",
                            default=None)
    metric_arg.add_argument("-pk", "--peaks",
                            dest="peaks",
                            type=str,
                            help="Full path and filename of the list with the indexed peaks' "
                                 "positions of the physiological data.",
                            default=None)
    metric_arg.add_argument("-tg", "--troughs",
                            dest="troughs",
                            type=str,
                            help="Full path and filename of the list with the indexed troughs' "
                                 "positions of the physiological data.",
                            default=None)
    metric_arg.add_argument("-os", "--oversampling",
                            dest="oversampling",
                            type=int,
                            help="Temporal oversampling factor. "
                                 "Default is 50.",
                            default=50)
    metric_arg.add_argument("-tl", "--time-length",
                            dest="time_length",
                            type=int,
                            help="RRF or CRF Kernel length in seconds.",
                            default=None)
    metric_arg.add_argument("-onset", "--onset",
                            dest="onset",
                            type=float,
                            help="Onset of the response in seconds. "
                                 "Default is 0.",
                            default=0)
    metric_arg.add_argument("-tr", "--tr",
                            dest="tr",
                            type=float,
                            help="TR of sequence in seconds.",
                            default=None)
    metric_arg.add_argument("-win", "--window",
                            dest="window",
                            type=int,
                            help="Size of the sliding window in seconds. "
                                 "Default is 6 seconds.",
                            default=6)
    metric_arg.add_argument("-lags", "--lags",
                            dest="lags",
                            nargs="*",
                            type=int,
                            help="List of lags to apply to the RV estimate "
                                 "in seconds.",
                            default=None)
    metric_arg.add_argument("-nscans", "--number-scans",
                            dest="nscans",
                            type=int,
                            help="Number of timepoints in the imaging data. "
                                 "Also called sub-bricks, TRs, scans, volumes."
                                 "Default is 1.",
                            default=1)
    metric_arg.add_argument("-nharm", "--number-harmonics",
                            dest="n_harm",
                            type=int,
                            help="Number of harmonics.",
                            default=None)

    # Other optional arguments
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
