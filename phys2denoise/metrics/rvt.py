import numpy as np
from scipy.interpolate import interp1d


def rvt(belt_ts, peaks, troughs, samplerate, lags=(0,4,8,12)):
    timestep = 1/samplerate
    time = np.array([i*timestep for i in range(len(belt_ts))])
    peak_vals = belt_ts[peaks]
    trough_vals = belt_ts[troughs]
    peak_time = time[peaks]
    mid_peak_time = (peak_time[:-1] + peak_time[1:])/2
    period = (peak_time[1:] - peak_time[:-1])
    trough_time = time[troughs]
    peak_interp = interp1d(peak_time, peak_vals, bounds_error=False, fill_value="extrapolate")(time)
    trough_interp = interp1d(trough_time, trough_vals, bounds_error=False, fill_value="extrapolate")(time)
    period_interp = interp1d(mid_peak_time, period, bounds_error=False, fill_value="extrapolate")(time)
    rvt = (peak_interp - trough_interp)/period_interp
    rvt_lags = np.zeros((len(rvt), len(lags)))
    for ind, lag in enumerate(lags):
        start_index = np.argmin(np.abs(time-lag))
        temp_rvt = np.concatenate((np.full((start_index), rvt[0]), rvt[:len(rvt)-start_index]))
        rvt_lags[:, ind] = temp_rvt

    return rvt_lags