"""Denoising metrics for chest belt recordings."""
import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.ndimage.filters import convolve1d
from scipy.signal import detrend, resample
from scipy.stats import zscore

mpl.use("TkAgg")
import matplotlib.pyplot as plt

from .. import references
from ..due import due
from . import utils


@due.dcite(references.POWER_2018)
def rpv(belt_ts, window):
    """Calculate respiratory pattern variability.

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
def env(belt_ts, samplerate, out_samplerate, window=10):
    """Calculate respiratory pattern variability across a sliding window.

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
    env_arr = (
        pd.Series(belt_ts).rolling(window=window, center=True).apply(rpv, window=window)
    )
    env_arr[np.isnan(env_arr)] = 0.0
    return env_arr


@due.dcite(references.CHANG_GLOVER_2009)
def rv(belt_ts, samplerate, out_samplerate, window=6, lags=(0,)):
    """Calculate respiratory variance.

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
        List of lags to apply to the rv estimate.
        Lags can be negative, zero, and/or positive.
        In seconds (like out_samplerate).

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
    rv_arr[np.isnan(rv_arr)] = 0.0

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


@due.dcite(references.CHANG_GLOVER_2009)
def rrf(samplerate, oversampling=50, time_length=50, onset=0.0, tr=2.0):
    """Calculate the respiratory response function using Chang and Glover's definition.

    Parameters
    ----------
    samplerate : :obj:`float`
        Sampling rate of data, in seconds.
    oversampling : :obj:`int`, optional
        Temporal oversampling factor. Default is 50.
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
    parameters, including oversampling, time_length, and onset, are modeled on
    nistats' HRF functions.

    References
    ----------
    .. [1] C. Chang & G. H. Glover, "Relationship between respiration,
       end-tidal CO2, and BOLD signals in resting-state fMRI," Neuroimage,
       issue 47, vol. 4, pp. 1381-1393, 2009.
    """

    def _rrf(t):
        rf = 0.6 * t ** 2.1 * np.exp(-t / 1.6) - 0.0023 * t ** 3.54 * np.exp(-t / 4.25)
        return rf

    dt = tr / oversampling
    time_stamps = np.linspace(
        0, time_length, np.rint(float(time_length) / dt).astype(np.int)
    )
    time_stamps -= onset
    rrf_arr = _rrf(time_stamps)
    rrf_arr = rrf_arr / max(abs(rrf_arr))
    return rrf_arr


def respiratory_phase(resp, sample_rate, n_scans, slice_timings, t_r):
    """Calculate respiratory phase from respiratory signal.

    Parameters
    ----------
    resp : 1D array_like
        Respiratory signal.
    sample_rate : float
        Sample rate of physio, in Hertz.
    n_scans
        Number of volumes in the imaging run.
    slice_timings
        Slice times, in seconds.
    t_r
        Sample rate of the imaging run, in seconds.

    Returns
    -------
    phase_resp : array_like
        Respiratory phase signal.
    """
    n_slices = np.shape(slice_timings)
    phase_resp = np.zeros((n_scans, n_slices))

    # generate histogram from respiratory signal
    # TODO: Replace with numpy.histogram
    resp_hist, resp_hist_bins = plt.hist(resp, bins=100)

    # first compute derivative of respiration signal
    resp_diff = np.diff(resp, n=1)

    for i_slice in range(n_slices):
        # generate slice acquisition timings across all scans
        times_crSlice = t_r * np.arange(n_scans) + slice_timings[i_slice]
        phase_resp_crSlice = np.zeros(n_scans)
        for j_scan in range(n_scans):
            iphys = int(
                max([1, round(times_crSlice[j_scan] * sample_rate)])
            )  # closest idx in resp waveform
            iphys = min([iphys, len(resp_diff)])  # cannot be longer than resp_diff
            thisBin = np.argmin(abs(resp[iphys] - resp_hist_bins))
            numerator = np.sum(resp_hist[0:thisBin])
            phase_resp_crSlice[j_scan] = (
                np.math.pi * np.sign(resp_diff[iphys]) * (numerator / len(resp))
            )

        phase_resp[:, i_slice] = phase_resp_crSlice

    return phase_resp
