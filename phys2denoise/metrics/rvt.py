import numpy as np
from scipy.interpolate import interp1d


def rvt(belt_ts, peaks, troughs, samplerate, window=10, lags=(0,)):
    timestep = 1/samplerate
    time = np.array([i*timestep for i in range(len(belt_ts))])
    peak_time = time[peaks]
    trough_time = time[troughs]
    peak_interp = interp1d(peak_time, belt_ts[peaks], bounds_error=False, fill_value="extrapolate")(time)
    trough_interp = interp1d(trough_time, belt_ts[troughs], bounds_error=False, fill_value="extrapolate")(time)
    rvt = peak_interp - trough_interp
    return rvt