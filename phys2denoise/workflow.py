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

from phys2denoise.cli.run import _get_parser
from phys2denoise.metrics.cardiac import crf
from phys2denoise.metrics.chest_belt import (
    respiratory_pattern_variability,
    respiratory_variance,
    respiratory_variance_time,
    rrf,
)
from phys2denoise.metrics.retroicor import retroicor

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


def select_input_args(metric, metric_args):
    """
    Retrieve required args for metric from a dictionary of possible arguments.

    This function checks what parameters are accepted by a metric.
    Then, for each parameter, check if the user provided it or not.
    If they did not, but the parameter is required, throw an error -
    unless it's "physio", reserved name for the timeseries input to a metric.
    Otherwise, use the default.

    Parameters
    ----------
    metric : function
        Metric function to retrieve arguments for
    metric_args : dict
        Dictionary containing all arguments for all functions requested by the
        user

    Returns
    -------
    args : dict
        Arguments to provide as input to metric

    Raises
    ------
    ValueError
        If a required argument is missing

    """
    args = {}

    # Check the parameters required by the metric and given by the user (see docstring)
    for param in signature(metric).parameters.values():
        if param.name not in metric_args:
            if param.default == _empty and param.name != "physio":
                raise ValueError(
                    f"Missing parameter {param} required " f"to run {metric}"
                )
            else:
                args[param.name] = param.default
        else:
            args[param.name] = metric_args[param.name]

    return args


@due.dcite(
    Doi(""),
    path="phys2denoise",
    description="Creation of regressors for physiological denoising",
    version=__version__,
    cite_module=True,
)
def phys2denoise(
    filename,
    outdir=".",
    metrics=[
        crf,
        respiratory_pattern_variability,
        respiratory_variance,
        respiratory_variance_time,
        rrf,
        "retroicor_card",
        "retroicor_resp",
    ],
    debug=False,
    quiet=False,
    **kwargs,
):
    """
    Run main workflow of phys2denoise.

    Runs the parser, does some checks on input, then computes the required metrics.

    Notes
    -----
    Any metric argument should go into kwargs!
    The code was greatly copied from phys2bids (copyright the physiopy community)

    """
    # Check options to make them internally coherent pt. I
    # #!# This can probably be done while parsing?
    outdir = os.path.abspath(outdir)
    log_path = os.path.join(outdir, "code", "logs")
    os.makedirs(log_path)

    # Create logfile name
    basename = "phys2denoise_"
    extension = "tsv"
    isotime = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
    logname = os.path.join(log_path, (basename + isotime + "." + extension))

    # Set logging format
    log_formatter = logging.Formatter(
        "%(asctime)s\t%(name)-12s\t%(levelname)-8s\t%(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Set up logging file and open it for writing
    log_handler = logging.FileHandler(logname)
    log_handler.setFormatter(log_formatter)
    sh = logging.StreamHandler()

    if quiet:
        logging.basicConfig(
            level=logging.WARNING,
            handlers=[log_handler, sh],
            format="%(levelname)-10s %(message)s",
        )
    elif debug:
        logging.basicConfig(
            level=logging.DEBUG,
            handlers=[log_handler, sh],
            format="%(levelname)-10s %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            handlers=[log_handler, sh],
            format="%(levelname)-10s %(message)s",
        )

    version_number = __version__
    LGR.info(f"Currently running phys2denoise version {version_number}")
    LGR.info(f"Input file is {filename}")

    # Check options to make them internally coherent pt. II
    # #!# This can probably be done while parsing?
    # filename, ftype = utils.check_input_type(filename)

    if not os.path.isfile(filename) and filename is not None:
        raise FileNotFoundError(f"The file {filename} does not exist!")

    # Read input file
    physio = np.genfromtxt(filename)

    # Prepare pandas dataset
    regr = pd.DataFrame()

    # Goes through the list of metrics and calls them
    for metric in metrics:
        if metric == "retroicor_card":
            args = select_input_args(retroicor, kwargs)
            args["card"] = True
            retroicor_regrs = retroicor(physio, **args)
            for vslice in range(len(args["slice_timings"])):
                for harm in range(args["n_harm"]):
                    key = f"rcor-card_s-{vslice}_hrm-{harm}"
                    regr[f"{key}_cos"] = retroicor_regrs[vslice][:, harm * 2]
                    regr[f"{key}_sin"] = retroicor_regrs[vslice][:, harm * 2 + 1]
        elif metric == "retroicor_resp":
            args = select_input_args(retroicor, kwargs)
            args["resp"] = True
            retroicor_regrs = retroicor(physio, **args)
            for vslice in range(len(args["slice_timings"])):
                for harm in range(args["n_harm"]):
                    key = f"rcor-resp_s-{vslice}_hrm-{harm}"
                    regr[f"{key}_cos"] = retroicor_regrs[vslice][:, harm * 2]
                    regr[f"{key}_sin"] = retroicor_regrs[vslice][:, harm * 2 + 1]
        else:
            args = select_input_args(metric, kwargs)
            regr[metric.__name__] = metric(physio, **args)

    # #!# Add regressors visualisation

    # Export regressors and sidecar
    out_filename = os.join(outdir, "derivatives", filename)
    regr.to_csv(out_filename, sep="\t", index=False, float_format="%.6e")
    # #!# Add sidecar export


def _main(argv=None):
    options = _get_parser().parse_args(argv)

    save_bash_call(options.outdir)

    phys2denoise(**vars(options))


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
