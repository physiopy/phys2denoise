"""Denoising metrics for chest belt recordings."""
import numpy as np
from scipy.interpolate import interp1d


def rvt(belt_ts, peaks, troughs, samplerate, lags=(0, 4, 8, 12)):
    """
    Implement the Respiratory Variance over Time (Birn et al. (2006)).

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
    time = np.array([i * timestep for i in range(len(belt_ts))])
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
