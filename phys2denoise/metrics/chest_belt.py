"""Denoising metrics for chest belt recordings
"""
import numpy as np
import pandas as pd
from scipy.ndimage.filters import convolve1d
from scipy.signal import resample, detrend
from scipy.stats import zscore

from .utils import apply_lags


def rpv(belt_ts):
    """Respiratory pattern variability

    1. Z-score respiratory belt signal
    2. Calculate upper envelope
    3. Calculate standard deviation of envelope
    """
    pass


def env(belt_ts, samplerate, out_samplerate, window=10):
    """Respiratory pattern variability calculated across a sliding window

    Across a sliding window, do the following:
    1. Z-score respiratory belt signal
    2. Calculate upper envelope
    3. Calculate standard deviation of envelope
    """
    pass


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
       issue 47, vol. 4, pp. 1381-1393, 2009.
    """
    # Raw respiratory variance
    rv_arr = pd.Series(belt_ts).rolling(window=window, center=True).std()
    rv_arr[np.isnan(rv_arr)] = 0.

    # Apply lags
    n_out_samples = int((belt_ts.shape[0] / samplerate) / out_samplerate)
    # convert lags from out_samplerate to samplerate
    delays = [abs(int(lag * samplerate)) for lag in lags]
    rv_with_lags = apply_lags(rv_arr, lags=delays)

    # Downsample to out_samplerate
    rv_with_lags = resample(rv_with_lags, num=n_out_samples, axis=0)

    # Convolve with rrf
    rrf_arr = rrf(out_samplerate)
    rv_convolved = convolve1d(rv_with_lags, rrf_arr, axis=0)

    # Concatenate the raw and convolved versions
    rv_combined = np.hstack((rv_with_lags, rv_convolved))

    # Detrend and normalize
    rv_combined = rv_combined - np.mean(rv_combined, axis=0)
    rv_combined = detrend(rv_combined, axis=0)
    rv_out = zscore(rv_combined, axis=0)
    return rv_out


def rvt(belt_ts, samplerate, out_samplerate, window=10):
    """Respiratory volume-per-time
    """
    pass


def rrf(samplerate):
    """Respiratory response function
    """
    pass
