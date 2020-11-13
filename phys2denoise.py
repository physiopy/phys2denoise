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
from copy import deepcopy
from shutil import copy as cp

import numpy as np

from phys2denoise import utils, _version
from phys2denoise.cli.run import _get_parser
from phys2denoise.metrics.cardiac import crf
from phys2denoise.metrics.chest_belt import rpv, rv, rvt, rrf
from phys2denoise.metrics.retroicor import compute_retroicor_regressors

from . import __version__
from .due import due, Doi

LGR = logging.getLogger(__name__)


@due.dcite(
     Doi(''),
     path='phys2denoise',
     description='Creation of regressors for physiological denoising',
     version=__version__,
     cite_module=True)
def phys2denoise(filename, outdir='.', metrics=[], debug=False, quiet=False):
    """
    Run main workflow of phys2denoise.

    Runs the parser, does some checks on input, then computes the required metrics.

    Notes
    -----
    The code was greatly copied from phys2bids (copyright the physiopy community)

    """
    # Check options to make them internally coherent pt. I
    # #!# This can probably be done while parsing?
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir)
    os.makedirs(os.path.join(outdir, 'code'))
    conversion_path = os.path.join(outdir, 'code', 'conversion')
    os.makedirs(conversion_path)

    # Create logfile name
    basename = 'phys2denoise_'
    extension = 'tsv'
    isotime = datetime.datetime.now().strftime('%Y-%m-%dT%H%M%S')
    logname = os.path.join(conversion_path, (basename + isotime + '.' + extension))

    # Set logging format
    log_formatter = logging.Formatter(
        '%(asctime)s\t%(name)-12s\t%(levelname)-8s\t%(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S')

    # Set up logging file and open it for writing
    log_handler = logging.FileHandler(logname)
    log_handler.setFormatter(log_formatter)
    sh = logging.StreamHandler()

    if quiet:
        logging.basicConfig(level=logging.WARNING,
                            handlers=[log_handler, sh], format='%(levelname)-10s %(message)s')
    elif debug:
        logging.basicConfig(level=logging.DEBUG,
                            handlers=[log_handler, sh], format='%(levelname)-10s %(message)s')
    else:
        logging.basicConfig(level=logging.INFO,
                            handlers=[log_handler, sh], format='%(levelname)-10s %(message)s')

    version_number = _version.get_versions()['version']
    LGR.info(f'Currently running phys2denoise version {version_number}')
    LGR.info(f'Input file is {filename}')

    # Save call.sh
    arg_str = ' '.join(sys.argv[1:])
    call_str = f'phys2denoise {arg_str}'
    f = open(os.path.join(conversion_path, 'call.sh'), "a")
    f.write(f'#!bin/bash \n{call_str}')
    f.close()

    # Check options to make them internally coherent pt. II
    # #!# This can probably be done while parsing?
    # filename, ftype = utils.check_input_type(filename)

    if not os.path.isfile(filename) and filename is not None:
        raise FileNotFoundError(f'The file {filename} does not exist!')

    # Read input file
    phys_in = np.genfromtxt(filename)

    # Goes through the list of metrics and calls them
    if not metrics:
        metrics = ['crf', 'rpv', 'rv', 'rvt', 'rrf', 'rcard', 'r']

    for 







def _main(argv=None):
    options = _get_parser().parse_args(argv)
    phys2denoise(**vars(options))


if __name__ == '__main__':
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
