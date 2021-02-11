"""Denoising metrics for cardio recordings."""
import numpy as np
from scipy import signal


def cpm(cardiac, i_hr, peaks, samplerate):
    """Calculate cardiac pulsatility model (CPM) regressors.

    Parameters
    ----------
    cardiac : 1D numpy.ndarray
        Raw PPG signal.
    i_hr : numpy.ndarray
        Instantaneous heart rate?
    peaks : 1D numpy.ndarray
        Index of PPG peaks.
    samplerate : float
        Sample rate in Hertz.

    Returns
    -------
    cpm_regressors
    cpm_amplitude_regressors
    retroicor_regressors

    Notes
    -----
    CPM was first developed in [1]_ and the original code is implemented in MATLAB.
    The original implementation, from which this code has been adapted, is released with an Apache
    2.0 license.
    More information about the CPM license can be found in the ``phys2denoise`` LICENSE.

    Here we log all meaningful changes made to this code from the original CPM code:
    1. Translation from MATLAB to Python.

    References
    ----------
    .. [1]: Kassinopoulos, M., & Mitsis, G. D. (2020).
            Physiological noise modeling in fMRI based on the pulsatile
            component of photoplethysmograph. bioRxiv.
            https://doi.org/10.1101/2020.06.01.128306
    """
    timestep = 1 / samplerate

    # number of seconds by which to shift each regressor
    SHIFT_CPM = 0.5
    SHIFT_CPM_VA = 0.5
    SHIFT_RETR = 0

    MODEL_ORDER = 8

    # Get filter parameters
    HIGH_PASS_FILTER_FREQ = 0.008  # in Hertz
    filt_b, filt_a = signal.butter(2, HIGH_PASS_FILTER_FREQ * 2 * timestep, "high")

    N_SECS_TO_REMOVE = 5  # delete first and last 5 seconds from output
    n_vals_to_remove = int(N_SECS_TO_REMOVE * samplerate)

    HRmean = np.mean(i_hr)  # mean heart-rate in beats-per-second?
    memory = 60 / HRmean  # average minutes-per-beat?

    cardiac_reduced = cardiac[n_vals_to_remove:-n_vals_to_remove]
    n_timepoints_reduced = len(cardiac_reduced)
    # high-pass filter the reduced cardiac signal
    cardiac_reduced = signal.filtfilt(filt_b, filt_a, cardiac_reduced)

    n_timepoints = len(cardiac)
    time = np.arange(0, n_timepoints * timestep, timestep)

    cpm_ir = func_CPM_cos(timestep, memory, MODEL_ORDER)

    # peaks in timeseries form (peaks are 1, all other timepoints are 0)
    peak_timeseries = np.zeros(time.shape)
    # peak amplitudes in timeseries form (peaks are amplitude value, all other timepoints are 0)
    peak_amplitudes = peak_timeseries.copy()
    for peak_time in peaks:
        time_distance = np.abs(time - peak_time)
        closest_time_idx = np.argmin(time_distance)
        peak_timeseries[closest_time_idx] = 1
        peak_amplitudes[closest_time_idx] = cardiac[closest_time_idx]

    cpm_regressors = np.zeros((n_timepoints, MODEL_ORDER * 2))
    cpm_amplitude_regressors = np.zeros((n_timepoints, MODEL_ORDER * 2))
    for i_col in range(MODEL_ORDER * 2):
        cpm_nonamplitudes = np.convolve(peak_timeseries, cpm_ir[:, i_col])
        cpm_nonamplitudes = cpm_nonamplitudes[:n_timepoints]
        cpm_regressors[:, i_col] = cpm_nonamplitudes
        cpm_amplitudes = np.convolve(peak_amplitudes, cpm_ir[:, i_col])
        cpm_amplitudes = cpm_amplitudes[:n_timepoints]
        cpm_amplitude_regressors[:, i_col] = cpm_amplitudes

    cpm_regressors = signal.filtfilt(filt_b, filt_a, cpm_regressors)
    cpm_amplitude_regressors = signal.filtfilt(filt_b, filt_a, cpm_amplitude_regressors)

    retroicor_regressors = RETR_Card_regressors_v2(time, peaks, MODEL_ORDER)
    retroicor_regressors = signal.filtfilt(filt_b, filt_a, retroicor_regressors)

    # Select relevant timepoints from regressors.
    idx = np.arange(n_timepoints_reduced)
    temp_idx = np.round(idx + n_vals_to_remove + (SHIFT_CPM * samplerate)).astype(int)
    cpm_regressors = cpm_regressors[temp_idx, :]

    temp_idx = np.round(idx + n_vals_to_remove + (SHIFT_CPM_VA * samplerate)).astype(int)
    cpm_amplitude_regressors = cpm_amplitude_regressors[temp_idx, :]

    temp_idx = np.round(idx + n_vals_to_remove + (SHIFT_RETR * samplerate)).astype(int)
    retroicor_regressors = retroicor_regressors[temp_idx, :]

    # Append ones?
    cpm_regressors = np.hstack((cpm_regressors, np.ones((n_timepoints_reduced, 1))))
    cpm_amplitude_regressors = np.hstack(
        (cpm_amplitude_regressors, np.ones((n_timepoints_reduced, 1)))
    )
    retroicor_regressors = np.hstack(
        (retroicor_regressors, np.ones((n_timepoints_reduced, 1)))
    )
    return cpm_regressors, cpm_amplitude_regressors, retroicor_regressors


def RETR_Card_regressors_v2(time, locsECG, M):
    """Calculate RETROICOR cardiac regressors.

    Parameters
    ----------
    time
    locsECG
    M

    Returns
    -------
    Regr
    """
    n_timepoints = len(time)
    Phi = np.zeros((n_timepoints))
    for i_time, timepoint in enumerate(time):
        minI = np.argmin(np.abs(locsECG - timepoint))

        minOnLeft = timepoint - locsECG[minI] > 0
        if (minI == 0) and not minOnLeft:
            t2 = locsECG[minI]
            t1 = t2 - 1
        elif (minI == (len(locsECG) - 1)) and minOnLeft:
            t1 = locsECG[minI]
            t2 = t1 + 1
        elif minOnLeft:
            t1 = locsECG[minI]
            t2 = locsECG[minI + 1]
        else:
            t1 = locsECG[minI - 1]
            t2 = locsECG[minI]

        Phi[i_time] = 2 * np.pi * (timepoint - t1) / (t2 - t1)

    Regr = np.zeros((n_timepoints, M * 2))
    for i in range(M):
        Regr[:, ((i - 1) * 2) + 1] = np.cos(i * Phi)
        Regr[:, i * 2] = np.sin(i * Phi)

    return Regr


def func_CPM_cos(timestep, memory, M):
    """Calculate CPM cosine values.

    Parameters
    ----------
    timestep : float
        Sample rate of cardiac recording, in seconds.
    memory : float
        ???
    M : int
        Model order?

    Returns
    -------
    IR_all : numpy.ndarray
        CPM metric at different model orders?
    """
    t_win = np.arange(0, memory, timestep)
    nIR = len(t_win)

    IR_all = np.zeros((nIR, M * 2))
    for m in range(M):
        IR_all[:, ((m - 1) * 2) + 1] = np.cos(m * 2 * np.pi * t_win / memory) - 1
        IR_all[:, ((m - 1) * 2) + 2] = np.sin(m * 2 * np.pi * t_win / memory)

    return IR_all
