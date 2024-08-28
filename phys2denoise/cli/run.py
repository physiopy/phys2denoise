# -*- coding: utf-8 -*-
"""Parser for phys2denoise."""


import argparse
import logging
import sys

import numpy as np
import pydra
from loguru import logger

from phys2denoise import __version__, tasks, workflow
from phys2denoise.metrics.cardiac import (
    cardiac_phase,
    heart_beat_interval,
    heart_rate,
    heart_rate_variability,
)
from phys2denoise.metrics.chest_belt import (
    env,
    respiratory_pattern_variability,
    respiratory_phase,
    respiratory_variance,
    respiratory_variance_time,
)
from phys2denoise.metrics.multimodal import retroicor
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
    required.add_argument(
        "-md",
        "--mode",
        dest="mode",
        type=str,
        help="Format of the input physiological data. Options are: "
        "physio or bids. Default is physio.",
    )

    # Important optional arguments
    optional = parser.add_argument_group("Optional arguments")
    optional.add_argument(
        "-out",
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
        const="respiratory_variance",
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
        const="respiratory_variance_time",
        help="Respiratory volume-per-time. Requires the following inputs: "
        "sample-rate, window, lags, peaks and troughs.",
        default=[],
    )
    resp_met.add_argument(
        "-rp",
        "--respiratory-phase",
        dest="metrics",
        action="append_const",
        const="respiratory_phase",
        help="Respiratory phase. Requires the following inputs: "
        "slice-timings, n_scans and t_r.",
        default=[],
    )

    card_met = parser.add_argument_group("Cardiac signal based metrics")
    card_met.add_argument(
        "-hrv",
        "--heart-rate-variability",
        dest="metrics",
        action="append_const",
        const="heart_rate_variability",
        help="Computes heart rate variability. Requires the following "
        "inputs: peaks, samplerate, window and central measure operator.",
        default=[],
    )
    card_met.add_argument(
        "-hbi",
        "--heart-beat-interval",
        dest="metrics",
        action="append_const",
        const="heart_beat_interval",
        help="Computes heart beat interval. Requires the following "
        "inputs: peaks, samplerate, window and central measure operator.",
        default=[],
    )
    card_met.add_argument(
        "-hr",
        "--heart-rate",
        dest="metrics",
        action="append_const",
        const="heart_rate",
        help="Computes heart rate. Requires the following "
        "inputs: peaks, samplerate, window and central measure operator.",
        default=[],
    )
    card_met.add_argument(
        "-cp",
        "--cardiac-phase",
        dest="metrics",
        action="append_const",
        const="cardiac_phase",
        help="Computes cardiac phase. Requires the following "
        "inputs: slice-timings, n_scans and t_r.",
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

    export_met = parser.add_argument_group("Export metrics")
    export_met.add_argument(
        "-e",
        "--exported-metrics",
        dest="metrics_to_export",
        nargs="+",
        type=str,
        help="Full path and filename of the list with the metrics to export.",
        default=None,
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
        "-tr",
        "--tr",
        dest="t_r",
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
        dest="n_scans",
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
    metric_arg.add_argument(
        "-sl",
        "--slice-timings",
        dest="slice_timings",
        nargs="*",
        type=float,
        help="Slice timings in seconds.",
        default=None,
    )

    # BIDS arguments
    bids = parser.add_argument_group("BIDS Arguments")
    bids.add_argument(
        "-sub",
        "--subject",
        dest="subject",
        type=str,
        help="Subject ID in BIDS format.",
        default=None,
    )
    bids.add_argument(
        "-ses",
        "--session",
        dest="session",
        type=str,
        help="Session ID in BIDS format.",
        default=None,
    )
    bids.add_argument(
        "-task",
        "--task",
        dest="task",
        type=str,
        help="Task ID in BIDS format.",
        default=None,
    )
    bids.add_argument(
        "-run",
        "--run",
        dest="run",
        type=str,
        help="Run ID in BIDS format.",
        default=None,
    )
    bids.add_argument(
        "-rec",
        "--recording",
        dest="recording",
        type=str,
        help="Recording ID in BIDS format.",
        default=None,
    )
    bids.add_argument(
        "-ch",
        "--channel",
        dest="bids_channel",
        type=str,
        help="Physiological signal channel ID in BIDS format.",
        default=None,
    )

    # Logging style
    log_style_group = parser.add_argument_group(
        "Logging style arguments (optional and mutually exclusive)",
        "Options to specify the logging style",
    )
    log_style_group_exclusive = log_style_group.add_mutually_exclusive_group()
    log_style_group_exclusive.add_argument(
        "-debug",
        "--debug",
        dest="debug",
        action="store_true",
        help="Print additional debugging info and error diagnostics to log file. Default is False.",
        default=False,
    )
    log_style_group_exclusive.add_argument(
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
    optional.add_argument(
        "-v", "--version", action="version", version=("%(prog)s " + __version__)
    )

    return parser


def main():
    """
    Main function to run the parser.

    Returns
    -------
    args : argparse dict
        Dictionary with all arguments parsed by the parser.
    """
    parser = _get_parser()
    args = parser.parse_args()
    LGR = logging.getLogger(__name__)
    LGR.setLevel(logging.DEBUG)

    logger.add(sys.stderr, level="DEBUG")

    logger.info(f"Running phys2denoise version: {__version__}")

    LGR.debug(f"Arguments: {args}")
    LGR.debug(f"Metrics to export: {args.metrics_to_export}")

    if args.metrics_to_export is None or args.metrics_to_export == "all":
        args.metrics_to_export = "all"

    bids_parameters = {
        "subject": args.subject,
        "session": args.session,
        "task": args.task,
        "run": args.run,
        "recording": args.recording,
    }

    # Conversions
    args.slice_timings = (
        np.array(args.slice_timings) if args.slice_timings is not None else None
    )

    metric_args = dict()
    for metric in args.metrics:
        metric_args[metric] = tasks.select_input_args(globals()[metric], vars(args))

    logger.debug(f"Metric args: {metric_args}")

    wf = workflow.build(
        input_file=args.filename,
        export_directory=args.outdir,
        metrics=args.metrics,
        metric_args=metric_args,
        metrics_to_export=args.metrics_to_export,
        mode=args.mode,
        fs=args.sample_rate,
        bids_parameters=bids_parameters,
        bids_channel=args.bids_channel,
        tr=args.t_r,
        debug=args.debug,
        quiet=args.quiet,
    )

    with pydra.Submitter(plugin="cf") as sub:
        sub(wf)

    wf()

    return wf.result().output.result


if __name__ == "__main__":
    main()
    # raise RuntimeError(
    #     "phys2denoise/cli/run.py should not be run directly;\n"
    #     "Please `pip install` phys2denoise and use the "
    #     "`phys2denoise` command"
    # )
