#!/usr/bin/env python3

"""
Phys2denoise is a python3 library meant to prepare physiological regressors for fMRI denoising.

The project is under development.

Copyright 2020, The physiopy community.
Please scroll to bottom to read full license.

"""

import datetime
import logging
import os
import sys
from inspect import _empty, signature

import numpy as np
import pandas as pd
from loguru import logger
from physutils.tasks import transform_to_physio
from pydra import Submitter, Workflow

import phys2denoise.tasks as tasks
from phys2denoise.cli.run import _get_parser
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

from . import __version__
from .due import Doi, due

LGR = logging.getLogger(__name__)
LGR.setLevel(logging.INFO)


def save_bash_call(outdir):
    """
    Save the bash call into file `p2d_call.sh`.

    Parameters
    ----------
    metric : function
        Metric function to retrieve arguments for
    metric_args : dict
        Dictionary containing all arguments for all functions requested by the
        user
    """
    arg_str = " ".join(sys.argv[1:])
    call_str = f"phys2denoise {arg_str}"
    outdir = os.path.abspath(outdir)
    log_path = os.path.join(outdir, "code", "logs")
    os.makedirs(log_path, exist_ok=True)
    isotime = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
    f = open(os.path.join(log_path, f"p2d_call_{isotime}.sh"), "a")
    f.write(f"#!bin/bash \n{call_str}")
    f.close()


def build(
    *,
    input_file,
    export_directory,
    metrics,
    metric_args,
    metrics_to_export,
    mode="auto",
    fs=None,
    bids_parameters=dict(),
    bids_channel=None,
    tr=None,
    debug=False,
    quiet=False,
    **kwargs,
) -> Workflow:
    # TODO: Use only loguru as the main logger, once the pydra/loguru issue is fixed
    logger.remove(0)
    if quiet:
        logger.add(
            sys.stderr,
            level="WARNING",
            colorize=True,
            backtrace=False,
            diagnose=False,
        )
        LGR.setLevel(logging.WARNING)
    elif debug:
        logger.add(
            sys.stderr,
            level="DEBUG",
            colorize=True,
            backtrace=True,
            diagnose=True,
        )
        LGR.setLevel(logging.DEBUG)
    else:
        logger.add(
            sys.stderr,
            level="INFO",
            colorize=True,
            backtrace=True,
            diagnose=False,
        )
        LGR.setLevel(logging.INFO)

    physio_file = os.path.abspath(input_file)
    export_directory = os.path.abspath(export_directory)

    wf = Workflow(
        name="metrics_wf",
        input_spec=[
            "phys",
            "fs",
            "mode",
            "metrics",
            "metric_args",
            "metrics_to_export",
            "bids_parameters",
            "bids_channel",
            "tr",
        ],
        phys=physio_file,
        fs=fs,
        mode=mode,
        metrics=metrics,
        metric_args=metric_args,
        bids_parameters=bids_parameters,
        bids_channel=bids_channel,
        metrics_to_export=metrics_to_export,
        tr=tr,
    )
    wf.add(
        transform_to_physio(
            name="transform_to_physio",
            input_file=wf.lzin.phys,
            fs=wf.lzin.fs,
            mode=wf.lzin.mode,
            bids_parameters=wf.lzin.bids_parameters,
            bids_channel=wf.lzin.bids_channel,
        )
    )
    wf.add(
        tasks.compute_metrics(
            name="compute_metrics",
            phys=wf.transform_to_physio.lzout.out,
            metrics=wf.lzin.metrics,
            args=wf.lzin.metric_args,
        )
    )
    wf.add(
        tasks.export_metrics(
            name="export_metrics",
            phys=wf.compute_metrics.lzout.out,
            metrics=wf.lzin.metrics_to_export,
            outdir=export_directory,
            tr=wf.lzin.tr,
        )
    )
    wf.set_output([("result", wf.compute_metrics.lzout.out)])

    return wf


def run(workflow: Workflow, plugin="cf", **plugin_args):
    with Submitter(plugin=plugin, **plugin_args) as sub:
        sub(workflow)

    return workflow.result()


@logger.catch()
@due.dcite(
    Doi(""),
    path="phys2denoise",
    description="Creation of regressors for physiological denoising",
    version=__version__,
    cite_module=True,
)
def phys2denoise():
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

    if args.debug:
        logger.add(
            sys.stderr,
            level="DEBUG",
            colorize=True,
            backtrace=True,
            diagnose=True,
        )
        LGR.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.add(
            sys.stderr,
            level="WARNING",
            colorize=True,
            backtrace=False,
            diagnose=False,
        )
        LGR.setLevel(logging.WARNING)
    else:
        logger.add(
            sys.stderr,
            level="INFO",
            colorize=True,
            backtrace=True,
            diagnose=False,
        )
        LGR.setLevel(logging.INFO)

    logger.info(f"Running phys2denoise version: {__version__}")

    LGR.debug(f"Arguments Provided: {args}")

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
    args.lags = np.array(args.lags) if args.lags is not None else None

    metric_args = dict()
    for metric in args.metrics:
        metric_args[metric] = tasks.select_input_args(globals()[metric], vars(args))

    logger.debug(f"Metrics: {args.metrics}")

    wf = build(
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

    with Submitter(plugin="cf") as sub:
        sub(wf)

    wf()

    return wf.result().output.result


def _main(argv=None):
    options = _get_parser().parse_args(argv)

    save_bash_call(options.outdir)

    phys2denoise()


if __name__ == "__main__":
    _main(sys.argv[1:])

"""
Copyright 2019, The phys2denoise community.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
