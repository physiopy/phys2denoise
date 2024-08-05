"""Miscellaneous utility functions for metric calculation."""

import logging

import numpy as np

from .. import references
from ..due import due

LGR = logging.getLogger(__name__)
LGR.setLevel(logging.INFO)


@due.dcite(references.SHMUELI_2007)
@due.dcite(references.CHANG_CUNNINGHAM_GLOVER_2009)
def crf(samplerate, time_length=32, onset=0.0, inverse=False):
    """
    Calculate the cardiac response function using Chang, Cunningham and Glover's definition.

    Parameters
    ----------
    samplerate : :obj:`float`
        Sampling rate of data, in seconds.
    time_length : :obj:`int`, optional
        RRF kernel length, in seconds. Default is 32.
    onset : :obj:`float`, optional
        Onset of the response, in seconds. Default is 0.
    inverse: `bool`, optional
        If True, return the additive inverse of the CRF, i.e. the iCRF.

    Returns
    -------
    crf : array-like
        Cardiac or "heart" response function

    Notes
    -----
    This cardiac response function was defined in [1]_, Appendix A.

    The core code for this function comes from metco2, while several of the
    parameters, including time_length, and onset, are modeled on
    nistats' HRF functions.

    References
    ----------
    .. [1] C. Chang, J. P. Cunnningham & G. H. Glover, "Influence of heartrate
       on the BOLD signal: The cardiac response function", NeuroImage,
       issue 47, vol. 4, pp. 857-869, 2009.
    .. [2] K.Shmueli and al., “Low-frequency fluctuations in the cardiac rate
       as a source of variance in the resting-state fMRI BOLD signal“,
       NeuroImage, issue 2, vol. 38, pp.306-320, 2007.
    """

    def _crf(t):
        rf = 0.6 * t**2.7 * np.exp(-t / 1.6) - 16 * (
            1 / np.sqrt(2 * np.pi * 9)
        ) * np.exp(-0.5 * (((t - 12) ** 2) / 9))
        return rf

    time_stamps = np.arange(0, time_length, 1 / samplerate)
    time_stamps -= onset
    crf_arr = _crf(time_stamps)
    crf_arr = crf_arr / max(abs(crf_arr))

    if inverse:
        return -crf_arr
    else:
        return crf_arr


@due.dcite(references.CHANG_CUNNINGHAM_GLOVER_2009)
@due.dcite(references.CHEN_2020)
def icrf(samplerate, time_length=32, onset=0.0):
    """
    Calculate the inverse of the cardiac response function.

    Parameters
    ----------
    samplerate : :obj:`float`
        Sampling rate of data, in seconds.
    time_length : :obj:`int`, optional
        RRF kernel length, in seconds. Default is 32.
    onset : :obj:`float`, optional
        Onset of the response, in seconds. Default is 0.
    inverse: `bool`, optional
        If True, return the additive inverse of the CRF, i.e. the iCRF.

    Returns
    -------
    icrf : array-like
        Inverse of cardiac or "heart" response function

    References
    ----------
    .. [1] C. Chang, J. P. Cunnningham & G. H. Glover, "Influence of heartrate
       on the BOLD signal: The cardiac response function", NeuroImage,
       issue 47, vol. 4, pp. 857-869, 2009.
    """
    return crf(samplerate, time_length=32, onset=0.0, inverse=True)


@due.dcite(references.CHANG_GLOVER_2009)
def rrf(samplerate, time_length=50, onset=0.0):
    """Calculate the respiratory response function using Chang and Glover's definition.

    Parameters
    ----------
    samplerate : :obj:`float`
        Sampling rate of data, in seconds..
    time_length : :obj:`int`, optional
        RRF kernel length, in seconds. Default is 50.
    onset : :obj:`float`, optional
        Onset of the response, in seconds. Default is 0.

    Returns
    -------
    rrf : array-like
        respiratory response function

    Notes
    -----
    This respiratory response function was defined in [1]_, Appendix A.

    The core code for this function comes from metco2, while several of the
    parameters, including time_length, and onset, are modeled on
    nistats' HRF functions.

    References
    ----------
    .. [1] C. Chang & G. H. Glover, "Relationship between respiration,
       end-tidal CO2, and BOLD signals in resting-state fMRI," NeuroImage,
       issue 47, vol. 4, pp. 1381-1393, 2009.
    """

    def _rrf(t):
        rf = 0.6 * t**2.1 * np.exp(-t / 1.6) - 0.0023 * t**3.54 * np.exp(-t / 4.25)
        return rf

    time_stamps = np.arange(0, time_length, 1 / samplerate)
    time_stamps -= onset
    rrf_arr = _rrf(time_stamps)
    rrf_arr = rrf_arr / max(abs(rrf_arr))
    return rrf_arr
