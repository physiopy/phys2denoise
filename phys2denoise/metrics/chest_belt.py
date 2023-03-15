"""Denoising metrics for chest belt recordings."""
import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.ndimage.filters import convolve1d
from scipy.signal import detrend
from scipy.stats import zscore
from scipy.interpolate import interp1d

mpl.use("TkAgg")
import matplotlib.pyplot as plt

from .. import references
from ..due import due
from . import utils


def rvt(belt_ts, peaks, troughs, samplerate, lags=(0, 4, 8, 12)):
    """
    Implement the Respiratory Variance over Time (Birn et al. 2006).

    Procedural choices influenced by RetroTS

    Parameters
    ----------
    belt_ts: array_like
        respiratory belt data - samples x 1
    peaks: array_like
        peaks found by peakdet algorithm
    troughs: array_like
        troughs found by peakdet algorithm
    samplerate: float
        sample rate in hz of respiratory belt
    lags: tuple
        lags in seconds of the RVT output. Default is 0, 4, 8, 12.

    Outputs
    -------
    rvt: array_like
        calculated RVT and associated lags.
    """
    timestep = 1 / samplerate
    # respiration belt timing
    time = np.arange(0, len(belt_ts) * timestep, timestep)
    peak_vals = belt_ts[peaks]
    trough_vals = belt_ts[troughs]
    peak_time = time[peaks]
    trough_time = time[troughs]
    mid_peak_time = (peak_time[:-1] + peak_time[1:]) / 2
    period = peak_time[1:] - peak_time[:-1]
    # interpolate peak values over all timepoints
    peak_interp = interp1d(
        peak_time, peak_vals, bounds_error=False, fill_value="extrapolate"
    )(time)
    # interpolate trough values over all timepoints
    trough_interp = interp1d(
        trough_time, trough_vals, bounds_error=False, fill_value="extrapolate"
    )(time)
    # interpolate period over  all timepoints
    period_interp = interp1d(
        mid_peak_time, period, bounds_error=False, fill_value="extrapolate"
    )(time)
    # full_rvt is (peak-trough)/period
    full_rvt = (peak_interp - trough_interp) / period_interp
    # calculate lags for RVT
    rvt_lags = np.zeros((len(full_rvt), len(lags)))
    for ind, lag in enumerate(lags):
        start_index = np.argmin(np.abs(time - lag))
        temp_rvt = np.concatenate(
            (
                np.full((start_index), full_rvt[0]),
                full_rvt[: len(full_rvt) - start_index],
            )
        )
        rvt_lags[:, ind] = temp_rvt

    return rvt_lags


@due.dcite(references.POWER_2018)
def rpv(resp, window=6):
    """Calculate respiratory pattern variability.

    Parameters
    ----------
    resp : 1D array_like
    window : int

    Returns
    -------
    rpv_val : float
        Respiratory pattern variability value.

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
    resp_z = zscore(resp)

    # Collect upper envelope
    rpv_upper_env = utils.rms_envelope_1d(resp_z, window)

    # Calculate standard deviation
    rpv_val = np.std(rpv_upper_env)
    return rpv_val


@due.dcite(references.POWER_2020)
def env(resp, samplerate, window=10):
    """Calculate respiratory pattern variability across a sliding window.

    Parameters
    ----------
    resp : (X,) :obj:`numpy.ndarray`
        A 1D array with the respiratory belt time series.
    samplerate : :obj:`float`
        Sampling rate for resp, in Hertz.
    window : :obj:`int`, optional
        Size of the sliding window, in seconds.
        Default is 10.

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
    # Convert window to Hertz
    window = int(window * samplerate)

    # Calculate RPV across a rolling window
    env_arr = (
        pd.Series(resp).rolling(window=window, center=True).apply(rpv, args=(window,))
    )
    env_arr[np.isnan(env_arr)] = 0.0
    return env_arr


@due.dcite(references.CHANG_GLOVER_2009)
def rv(resp, samplerate, window=6):
    """Calculate respiratory variance.

    Parameters
    ----------
    resp : (X,) :obj:`numpy.ndarray`
        A 1D array with the respiratory belt time series.
    samplerate : :obj:`float`
        Sampling rate for resp, in Hertz.
    window : :obj:`int`, optional
        Size of the sliding window, in seconds.
        Default is 6.

    Returns
    -------
    rv_out : (X, 2) :obj:`numpy.ndarray`
        Respiratory variance values.
        The first column is raw RV values, after detrending/normalization.
        The second column is RV values convolved with the RRF, after detrending/normalization.

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
    # Convert window to Hertz
    window = int(window * samplerate)

    # Raw respiratory variance
    rv_arr = pd.Series(resp).rolling(window=window, center=True).std()
    rv_arr[np.isnan(rv_arr)] = 0.0

    # Convolve with rrf
    rrf_arr = rrf(samplerate, oversampling=1)
    rv_convolved = convolve1d(rv_arr, rrf_arr, axis=0)

    # Concatenate the raw and convolved versions
    rv_combined = np.stack((rv_arr, rv_convolved), axis=-1)

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
        Respiratory phase signal, of shape (n_scans, n_slices).
    """
    assert slice_timings.ndim == 1, "Slice times must be a 1D array"
    n_slices = np.size(slice_timings)
    phase_resp = np.zeros((n_scans, n_slices))

    # generate histogram from respiratory signal
    # TODO: Replace with numpy.histogram
    resp_hist, resp_hist_bins, _ = plt.hist(resp, bins=100)

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
