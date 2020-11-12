# -*- coding: utf-8 -*-
"""Parser for phys2denoise."""


import argparse

from phys2denoise import __version__


def _get_parser():
    """
    Parse command line inputs for this function.

    Returns
    -------
    parser.parse_args() : argparse dict

    Notes
    -----
    # Argument parser follow template provided by RalphyZ.
    # https://stackoverflow.com/a/43456577
    """
    parser = argparse.ArgumentParser()
    optional = parser._action_groups.pop()
    metric = parser._action_groups.pop()
    required = parser.add_argument_group('Required Argument:')
    required.add_argument('-in', '--input-file',
                          dest='filename',
                          type=str,
                          help='Path/name of the file containing physiological '
                               'data, with or without extension.',
                          required=True)
    metric.add_argument('-crf', '--crf',
                        dest='metric_list',
                        action='append_const',
                        const='crf',
                        help='Cardiac response function. Needs the following '
                             'inputs:sr, os, tl, onset and tr.',
                        default=False)
    metric.add_argument('-rpv', '--rpv',
                        dest='metric_list',
                        action='append_const',
                        const='rpv',
                        help='<span class="x x-first x-last">Respiratory</span> pattern variability. Needs the following '
                             'inputs: bts and win.',
                        default=False)
    metric.add_argument('-env', '--env',
                        dest='metric_list',
                        action='append_const',
                        const='env',
                        help='Respiratory pattern variability calculated across a sliding '
                             'window. Needs the following inputs: bts, sr, osr, win and lags.',
                        default=False)
    metric.add_argument('-rv', '--rv',
                        dest='metric_list',
                        action='append_const',
                        const='rv',
                        help='Respiratory variance. Needs the following inputs: '
                             'bts, sr, osr, win and lags.',
                        default=False)
    metric.add_argument('-rvt', '--rvt',
                        dest='metric_list',
                        action='append_const',
                        const='rvt',
                        help='Respiratory volume-per-time. Needs the following inputs: '
                             'bts, sr, osr, win and lags.',
                        default=False)
    metric.add_argument('-rrf', '--rrf',
                        dest='metric_list',
                        action='append_const',
                        const='rrf',
                        help='Respiratory response function. Needs the following inputs: '
                             'sr, os, tl, onset and tr.',
                        default=False)
    metric.add_argument('-rcard', '--retroicor-card',
                        dest='metric_list',
                        action='append_const',
                        const='r_card',
                        help='Computes regressors for cardiac signal. Needs the following '
                             'inputs: tr, nscans, slt and n_harm.',
                        default=False)
    metric.add_argument('-rresp', '--retroicor-resp',
                        dest='metric_list',
                        action='append_const',
                        const='r_resp',
                        help='Computes regressors for respiratory signal. Needs the following  '
                             'inputs: tr, nscans, slt and n_harm.',
                        default=False)
    optional.add_argument('-outdir', '--output-dir',
                          dest='outdir',
                          type=str,
                          help='Folder where output should be placed. '
                               'Default is current folder.',
                          default='.')
    optional.add_argument('-sr', '--sample-rate',
                          dest='sample_rate',
                          type=float,
                          help='Sampling rate of the physiological data in Hz.',
                          default=None)
    optional.add_argument('-pk', '--peaks',
                          dest='peaks',
                          type=str,
                          help='Filename of the list with the indexed peaks\' positions'
                               ' of the physiological data.',
                          default=None)
    optional.add_argument('-thr', '--throughts',
                          dest='throughts',
                          type=str,
                          help='Filename of the list with the indexed peaks\' positions'
                               ' of the physiological data.',
                          default=None)
    optional.add_argument('-os', '--oversampling',
                          dest='oversampling',
                          type=int,
                          help='Temporal oversampling factor in seconds. '
                               'Default is 50.',
                          default=50)
    optional.add_argument('-tl', '--time-length',
                          dest='time_length',
                          type=int,
                          help='RRF Kernel length in seconds.',
                          default=None)
    optional.add_argument('-onset', '--onset',
                          dest='onset',
                          type=float,
                          help='Onset of the response in seconds. '
                               'Default is 0.',
                          default=0)
    optional.add_argument('-tr', '--tr',
                          dest='tr',
                          type=float,
                          help='TR of sequence in seconds.',
                          default=None)
    optional.add_argument('-bts', '--belt-ts',
                          dest='belt_ts',
                          type=str,
                          help='Filename of the 1D array containing the .'
                               'respiratory belt time series.',
                          default=None)
    optional.add_argument('-win', '--window',
                          dest='window',
                          type=int,
                          help='Size of the sliding window in seconds. '
                               'Default is 6 seconds.',
                          default=6)
    optional.add_argument('-osr', '--out-samplerate',
                          dest='out_samplerate',
                          type=float,
                          help='Sampling rate for the output time series '
                               'in seconds. Corresponds to TR in fMRI data.',
                          default=None)
    optional.add_argument('-lags', '--lags',
                          dest='lags',
                          nargs='*',
                          type=int,
                          action='append',
                          help='List of lags to apply to the rv estimate '
                               'in seconds.',
                          default=None)
    optional.add_argument('-nscans', '--nscans',
                          dest='nscans',
                          type=int,
                          help='Number of scans. Default is 1.',
                          default=1)
    optional.add_argument('-slt', '--slice-timings',
                          dest='slice_timings',
                          type=str,
                          help='Filename with the slice timings.',
                          default=None)
    optional.add_argument('-nharm', '--number-harmonics',
                          dest='n_harm',
                          type=int,
                          help='Number of harmonics. ',
                          default=None)
    optional.add_argument('-debug', '--debug',
                          dest='debug',
                          action='store_true',
                          help='Only print debugging info to log file. Default is False.',
                          default=False)
    optional.add_argument('-quiet', '--quiet',
                          dest='quiet',
                          action='store_true',
                          help='Only print warnings to log file. Default is False.',
                          default=False)
    optional.add_argument('-v', '--version', action='version',
                          version=('%(prog)s ' + __version__))

    parser._action_groups.append(optional)

    return parser


if __name__ == '__main__':
    raise RuntimeError('phys2denoise/cli/run.py should not be run directly;\n'
                       'Please `pip install` phys2denoise and use the '
                       '`phys2denoise` command')
