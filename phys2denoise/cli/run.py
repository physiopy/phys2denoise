# -*- coding: utf-8 -*-
"""Parser for phys2denoise."""


import argparse

from phys2denoise import __version__
from phys2denoise.metrics.cardiac import heart_beat_interval, heart_rate_variability
from phys2denoise.metrics.chest_belt import (
    env,
    respiratory_pattern_variability,
    respiratory_variance,
    respiratory_variance_time,
)
from phys2denoise.metrics.responses import crf, icrf, rrf


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

    parser = argparse.ArgumentParser(
        description=(
            "%(prog)s, a tool to create physiological-based metrics to denoise "
            "functional neuroimaging data. It generates regressors to be used in "
            "other softwares.\n"
            f"Version {__version__}"
        ),
        add_help=False,
    )
    # Required arguments
    required = parser.add_argument_group("Required Argument")
    required.add_argument(
        "-in",
        "--input-file",
        dest="filename",
        type=str,
        help="Full path and name of the file containing "
        "physiological data, with or without extension.",
        required=True,
    )

    # Important optional arguments
    optional = parser.add_argument_group("Optional arguments")
    optional.add_argument(
        "-outdir",
        "--output-dir",
        dest="outdir",
        type=str,
        help="Folder where output should be placed. Default is current folder.",
        default=".",
    )

    # Metric selection
    resp_met = parser.add_argument_group("Respiratory effort based metrics")
    resp_met.add_argument(
        "-rpv",
        "--respiratory-pattern-variability",
        dest="metrics",
        action="append_const",
        const=respiratory_pattern_variability,
        help="Respiratory pattern variability. Requires the following "
        "input: window.",
        default=[],
    )
    resp_met.add_argument(
        "-env",
        "--envelope",
        dest="metrics",
        action="append_const",
        const=env,
        help="Respiratory pattern variability calculated across a sliding "
        "window. Requires the following inputs: sample-rate, window and lags.",
        default=[],
    )
    resp_met.add_argument(
        "-rv",
        "--respiratory-variance",
        dest="metrics",
        action="append_const",
        const=respiratory_variance,
        help="Respiratory variance. Requires the following inputs: "
        "sample-rate, window and lags. If the input file "
        "not a .phys file, it also requires peaks and troughs",
        default=[],
    )
    resp_met.add_argument(
        "-rvt",
        "--respiratory-variance-per-time",
        dest="metrics",
        action="append_const",
        const=respiratory_variance_time,
        help="Respiratory volume-per-time. Requires the following inputs: "
        "sample-rate, window, lags, peaks and troughs.",
        default=[],
    )

    card_met = parser.add_argument_group("Cardiac signal based metrics")
    card_met.add_argument(
        "-hrv",
        "--heart-rate-variability",
        dest="metrics",
        action="append_const",
        const=heart_rate_variability,
        help="Computes heart rate variability. Requires the following "
        "inputs: peaks, samplerate, window and central measure operator.",
        default=[],
    )
    card_met.add_argument(
        "-hbi",
        "--heart-beat-interval",
        dest="metrics",
        action="append_const",
        const=heart_beat_interval,
        help="Computes heart beat interval. Requires the following "
        "inputs: peaks, samplerate, window and central measure operator.",
        default=[],
    )

    mmod_met = parser.add_argument_group("Multimodal signals based metrics")
    mmod_met.add_argument(
        "-rcard",
        "--retroicor-card",
        dest="metrics",
        action="append_const",
        const="r_card",
        help="RETROICOR regressors for cardiac signal. Requires the following "
        "inputs: tr, nscans and n_harm.",
        default=[],
    )
    mmod_met.add_argument(
        "-rresp",
        "--retroicor-resp",
        dest="metrics",
        action="append_const",
        const="r_resp",
        help="RETROICOR regressors for respiratory signal. Requires the following  "
        "inputs: tr, nscans and n_harm.",
        default=[],
    )

    rfs = parser.add_argument_group("Response Functions")
    rfs.add_argument(
        "-crf",
        "--cardiac-response-function",
        dest="metrics",
        action="append_const",
        const=crf,
        help="Cardiac response function. Requires the following "
        "inputs: sample-rate, time-length, and onset.",
        default=[],
    )
    rfs.add_argument(
        "-icrf",
        "--inverse-cardiac-response-function",
        dest="metrics",
        action="append_const",
        const=icrf,
        help="Inverse of the cardiac response function. Requires the following "
        "inputs: sample-rate, time-length, and onset.",
        default=[],
    )
    rfs.add_argument(
        "-rrf",
        "--respiratory-response-function",
        dest="metrics",
        action="append_const",
        const=rrf,
        help="Respiratory response function. Requires the following inputs: "
        "sample-rate, time-length, and onset.",
        default=[],
    )

    # Metric arguments
    metric_arg = parser.add_argument_group("Metrics Arguments")
    metric_arg.add_argument(
        "-sr",
        "--sample-rate",
        dest="sample_rate",
        type=float,
        help="Sampling rate of the physiological data in Hz.",
        default=None,
    )
    metric_arg.add_argument(
        "-pk",
        "--peaks",
        dest="peaks",
        type=str,
        help="Full path and filename of the list with the indexed peaks' "
        "positions of the physiological data.",
        default=None,
    )
    metric_arg.add_argument(
        "-tg",
        "--troughs",
        dest="troughs",
        type=str,
        help="Full path and filename of the list with the indexed troughs' "
        "positions of the physiological data.",
        default=None,
    )
    metric_arg.add_argument(
        "-cmo",
        "--central-measure-operator",
        dest="central_measure",
        type=str,
        help='Central measure operator to use in cardiac metrics. Default is "mean."',
        default="mean",
    )
    metric_arg.add_argument(
        "-tl",
        "--time-length",
        dest="time_length",
        type=int,
        help="RRF or CRF Kernel length in seconds.",
        default=None,
    )
    metric_arg.add_argument(
        "-onset",
        "--onset",
        dest="onset",
        type=float,
        help="Onset of the response in seconds. Default is 0.",
        default=0,
    )
    metric_arg.add_argument(
        "-tr",
        "--tr",
        dest="tr",
        type=float,
        help="TR of sequence in seconds.",
        default=None,
    )
    metric_arg.add_argument(
        "-win",
        "--window",
        dest="window",
        type=int,
        help="Size of the sliding window in seconds. Default is 6 seconds.",
        default=6,
    )
    metric_arg.add_argument(
        "-lags",
        "--lags",
        dest="lags",
        nargs="*",
        type=int,
        help="List of lags to apply to the RV estimate in seconds.",
        default=None,
    )
    metric_arg.add_argument(
        "-nscans",
        "--number-scans",
        dest="nscans",
        type=int,
        help="Number of timepoints in the imaging data. "
        "Also called sub-bricks, TRs, scans, volumes."
        "Default is 1.",
        default=1,
    )
    metric_arg.add_argument(
        "-nharm",
        "--number-harmonics",
        dest="n_harm",
        type=int,
        help="Number of harmonics.",
        default=None,
    )

    # Other optional arguments
    otheropt = parser.add_argument_group("Other optional arguments")
    otheropt.add_argument(
        "-debug",
        "--debug",
        dest="debug",
        action="store_true",
        help="Only print debugging info to log file. Default is False.",
        default=False,
    )
    otheropt.add_argument(
        "-quiet",
        "--quiet",
        dest="quiet",
        action="store_true",
        help="Only print warnings to log file. Default is False.",
        default=False,
    )
    optional.add_argument(
        "-h", "--help", action="help", help="Show this help message and exit"
    )
    otheropt.add_argument(
        "-v", "--version", action="version", version=("%(prog)s " + __version__)
    )

    return parser


if __name__ == "__main__":
    raise RuntimeError(
        "phys2denoise/cli/run.py should not be run directly;\n"
        "Please `pip install` phys2denoise and use the "
        "`phys2denoise` command"
    )
