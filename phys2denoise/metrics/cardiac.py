"""Denoising metrics for cardio recordings."""
import numpy as np

from .. import references
from ..due import due


def iht():
    """Calculate instantaneous heart rate."""
    pass


@due.dcite(references.CHANG_GLOVER_2009)
def crf(samplerate, oversampling=50, time_length=32, onset=0.0, tr=2.0):
    """Calculate the cardiac response function using Chang and Glover's definition.

    Parameters
    ----------
    samplerate : :obj:`float`
        Sampling rate of data, in seconds.
    oversampling : :obj:`int`, optional
        Temporal oversampling factor, in seconds. Default is 50.
    time_length : :obj:`int`, optional
        RRF kernel length, in seconds. Default is 32.
    onset : :obj:`float`, optional
        Onset of the response, in seconds. Default is 0.

    Returns
    -------
    crf : array-like
        Cardiac or "heart" response function

    Notes
    -----
    This cardiac response function was defined in [1]_, Appendix A.

    The core code for this function comes from metco2, while several of the
    parameters, including oversampling, time_length, and onset, are modeled on
    nistats' HRF functions.

    References
    ----------
    .. [1] C. Chang & G. H. Glover, "Relationship between respiration,
       end-tidal CO2, and BOLD signals in resting-state fMRI," Neuroimage,
       issue 47, vol. 4, pp. 1381-1393, 2009.
    """

    def _crf(t):
        rf = 0.6 * t ** 2.7 * np.exp(-t / 1.6) - 16 * (
            1 / np.sqrt(2 * np.pi * 9)
        ) * np.exp(-0.5 * (((t - 12) ** 2) / 9))
        return rf

    dt = tr / oversampling
    time_stamps = np.linspace(
        0, time_length, np.rint(float(time_length) / dt).astype(np.int)
    )
    time_stamps -= onset
    crf_arr = _crf(time_stamps)
    crf_arr = crf_arr / max(abs(crf_arr))
    return crf_arr
