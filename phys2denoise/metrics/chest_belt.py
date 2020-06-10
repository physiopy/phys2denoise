"""Denoising metrics for chest belt recordings
"""
import numpy as np
import pandas as pd
from scipy.ndimage.filters import convolve1d
from scipy.signal import resample, detrend
from scipy.stats import zscore

from . import utils
from ..due import due
from .. import references


@due.dcite(references.POWER_2018)
def rpv(belt_ts, window):
    """Respiratory pattern variability

    Parameters
    ----------
    belt_ts
    window

    Returns
    -------
    rpv_arr

    Notes
    -----
    This metric was first introduced in [1]_.

    1. Z-score respiratory belt signal
    2. Calculate upper envelope
    3. Calculate standard deviation of envelope

    References
    ----------
    .. [1] J. D. Power et al., "Ridding fMRI data of motion-related influences:
       Removal of signals with distinct spatial and physical bases in multiecho
       data," Proceedings of the National Academy of Sciences, issue 9, vol.
       115, pp. 2105-2114, 2018.
    """
    # First, z-score respiratory traces
    resp_z = zscore(belt_ts)

    # Collect upper envelope
    rpv_upper_env = utils.rms_envelope_1d(resp_z, window)

    # Calculate standard deviation
    rpv_arr = np.std(rpv_upper_env)
    return rpv_arr


@due.dcite(references.POWER_2020)
def env(belt_ts, samplerate, out_samplerate, window=10, lags=(0,)):
    """Respiratory pattern variability calculated across a sliding window

    Parameters
    ----------
    belt_ts : (X,) :obj:`numpy.ndarray`
        A 1D array with the respiratory belt time series.
    samplerate : :obj:`float`
        Sampling rate for belt_ts, in Hertz.
    out_samplerate : :obj:`float`
        Sampling rate for the output time series in seconds.
        Corresponds to TR in fMRI data.
    window : :obj:`int`, optional
        Size of the sliding window, in the same units as out_samplerate.
        Default is 6.
    lags : (Y,) :obj:`tuple` of :obj:`int`, optional
        List of lags to apply to the rv estimate. In the same units as
        out_samplerate.

    Returns
    -------
    env_arr

    Notes
    -----
    This metric was first introduced in [1]_.

    Across a sliding window, do the following:
    1. Z-score respiratory belt signal
    2. Calculate upper envelope
    3. Calculate standard deviation of envelope

    References
    ----------
    .. [1] J. D. Power et al., "Characteristics of respiratory measures in
       young adults scanned at rest, including systematic changes and 'missed'
       deep breaths," Neuroimage, vol. 204, 2020.
    """
    window = window * samplerate / out_samplerate
    # Calculate RPV across a rolling window
    env_arr = pd.Series(belt_ts).rolling(window=window, center=True).apply(
        rpv, window=window)
    env_arr[np.isnan(env_arr)] = 0.
    return env_arr


@due.dcite(references.CHANG_GLOVER_2009)
def rv(belt_ts, samplerate, out_samplerate, window=6, lags=(0,)):
    """Respiratory variance

    Parameters
    ----------
    belt_ts : (X,) :obj:`numpy.ndarray`
        A 1D array with the respiratory belt time series.
    samplerate : :obj:`float`
        Sampling rate for belt_ts, in Hertz.
    out_samplerate : :obj:`float`
        Sampling rate for the output time series in seconds.
        Corresponds to TR in fMRI data.
    window : :obj:`int`, optional
        Size of the sliding window, in the same units as out_samplerate.
        Default is 6.
    lags : (Y,) :obj:`tuple` of :obj:`int`, optional
        List of lags to apply to the rv estimate. In the same units as
        out_samplerate.

    Returns
    -------
    rv_out : (Z, 2Y) :obj:`numpy.ndarray`
        Respiratory variance values, with lags applied, downsampled to
        out_samplerate, convolved with an RRF, and detrended/normalized.
        The first Y columns are not convolved with the RRF, while the second Y
        columns are.

    Notes
    -----
    Respiratory variance (RV) was introduced in [1]_, and consists of the
    standard deviation of the respiratory trace within a 6-second window.

    This metric is often lagged back and/or forward in time and convolved with
    a respiratory response function before being included in a GLM.
    Regressors also often have mean and linear trends removed and are
    standardized prior to regressions.

    References
    ----------
    .. [1] C. Chang & G. H. Glover, "Relationship between respiration,
       end-tidal CO2, and BOLD signals in resting-state fMRI," Neuroimage,
       issue 4, vol. 47, pp. 1381-1393, 2009.
    """
    # Raw respiratory variance
    rv_arr = pd.Series(belt_ts).rolling(window=window, center=True).std()
    rv_arr[np.isnan(rv_arr)] = 0.

    # Apply lags
    n_out_samples = int((belt_ts.shape[0] / samplerate) / out_samplerate)
    # convert lags from out_samplerate to samplerate
    delays = [abs(int(lag * samplerate)) for lag in lags]
    rv_with_lags = utils.apply_lags(rv_arr, lags=delays)

    # Downsample to out_samplerate
    rv_with_lags = resample(rv_with_lags, num=n_out_samples, axis=0)

    # Convolve with rrf
    rrf_arr = rrf(out_samplerate, oversampling=1)
    rv_convolved = convolve1d(rv_with_lags, rrf_arr, axis=0)

    # Concatenate the raw and convolved versions
    rv_combined = np.hstack((rv_with_lags, rv_convolved))

    # Detrend and normalize
    rv_combined = rv_combined - np.mean(rv_combined, axis=0)
    rv_combined = detrend(rv_combined, axis=0)
    rv_out = zscore(rv_combined, axis=0)
    return rv_out


def rvt(belt_ts, samplerate, out_samplerate, window=10, lags=(0,)):
    """Respiratory volume-per-time
    """
    pass


@due.dcite(references.CHANG_GLOVER_2009)
def rrf(samplerate, oversampling=50, time_length=50, onset=0., tr=2.):
    """
    Calculate the respiratory response function using the definition
    supplied in Chang and Glover (2009).

    Inputs
    ------
    samplerate : :obj:`float`
        Sampling rate of data, in seconds.
    oversampling : :obj:`int`, optional
        Temporal oversampling factor, in seconds. Default is 50.
    time_length : :obj:`int`, optional
        RRF kernel length, in seconds. Default is 50.
    onset : :obj:`float`, optional
        Onset of the response, in seconds. Default is 0.

    Outputs
    -------
    rrf: array-like
        respiratory response function

    Notes
    -----
    This respiratory response function was defined in [1]_, Appendix A.

    The core code for this function comes from metco2, while several of the
    parameters, including oversampling, time_length, and onset, are modeled on
    nistats' HRF functions.

    References
    ----------
    .. [1] C. Chang & G. H. Glover, "Relationship between respiration,
       end-tidal CO2, and BOLD signals in resting-state fMRI," Neuroimage,
       issue 47, vol. 4, pp. 1381-1393, 2009.
    """
    def _rrf(t):
        rf = (0.6 * t ** 2.1 * np.exp(-t / 1.6) - 0.0023 * t ** 3.54 * np.exp(-t / 4.25))
        return rf
    dt = tr / oversampling
    time_stamps = np.linspace(0, time_length,
                              np.rint(float(time_length) / dt).astype(np.int))
    time_stamps -= onset
    rrf_arr = _rrf(time_stamps)
    rrf_arr = rrf_arr / max(abs(rrf_arr))
    return rrf_arr
