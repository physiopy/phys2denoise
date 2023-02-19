"""Denoising metrics for cardio recordings."""
import numpy as np

from .responses import crf


def iht():
    """Calculate instantaneous heart rate."""
    pass


def heart_beat_interval(card, peaks, samplerate, window, central_measure="mean"):
    """Calculate the average heart beats interval (HBI) in a sliding window.

    Parameters
    ----------
    card : str or 1D numpy.ndarray
        Timeseries of recorded cardiac signal
    peaks : str or 1D numpy.ndarray
        array of peak indexes for card.
    samplerate : float
        Sampling rate for card, in Hertz.
    window : float, optional
        Size of the sliding window, in seconds.
        Default is 6.
    central_measure : string
        Measure of the center used (mean or median).
        Default is "mean".
    Returns
    -------
    hbi : 2D numpy.ndarray
        Heart Beats Interval values.
        The first column is raw HBI values.
        The second column is HBI values convolved with the RRF.

    Notes
    -----
    Heart beats interval (HBI) was introduced in [1]_, and consists of the
    average of the time interval between two heart beats based on ppg data within
    a 6-second window.

    This metric is often lagged back and/or forward in time and convolved with
    an inverse of the cardiac response function before being included in a GLM.

    References
    ----------
    .. [1] J. E. Chen & L. D. Lewis, "Resting-state "physiological networks"", Neuroimage,
        vol. 213, pp. 116707, 2020.
    """
    # Convert window to time points
    size = len(card)
    halfwindow_samples = int(round(window * samplerate / 2))
    hbi_arr = np.empty(size)

    for i in range(size):
        if i < 120:
            window_tp = i + halfwindow_samples
        elif i > (size - 1 - 120):
            window_tp = (size - 1 - i) + halfwindow_samples
        else:
            window_tp = window_size

        condition = (card.peaks >= (i - window_tp / 2)) & (card.peaks <= (i + window_tp / 2))
        peaks = card.peaks[condition]

        if central_measure == "mean":
            hbi_arr[i] = np.mean(np.ediff1d(peaks))
        elif central_measure == "median":
            hbi_arr[i] = np.median(np.ediff1d(peaks))
        else:
            return

    hbi_arr[np.isnan(hbi_arr)] = 0.0

    # Convolve with crf
    crf_arr = crf(samplerate)
    icrf_arr = -crf_arr  # geometric mean
    hbi_convolved = np.convolve(hbi_arr, icrf_arr)
    hbi_convolved = np.interp(hbi_convolved, (hbi_convolved.min(), hbi_convolved.max()), (hbi_arr.min(), hbi_arr.max()))
    #hbi_convolved = convolve1d(hbi_arr, icrf_arr, axis=0)

    # Concatenate the raw and convolved versions
    hbi_combined = np.stack((hbi_arr, hbi_convolved), axis=-1)

    # Normalize to z-score
    hbi_combined = hbi_combined - np.mean(hbi_combined, axis=0)
    hbi_out = zscore(hbi_combined, axis=0)
    return hbi_out


def cardiac_phase(peaks, sample_rate, slice_timings, n_scans, t_r):
    """Calculate cardiac phase from cardiac peaks.

    Assumes that timing of cardiac events are given in same units
    as slice timings, for example seconds.

    Parameters
    ----------
    peaks : 1D array_like
        Cardiac peak times, in seconds.
    sample_rate : float
        Sample rate of physio, in Hertz.
    slice_timings : 1D array_like
        Slice times, in seconds.
    n_scans : int
        Number of volumes in the imaging run.
    t_r : float
        Sampling rate of the imaging run, in seconds.

    Returns
    -------
    phase_card : array_like
        Cardiac phase signal, of shape (n_scans,)
    """
    assert slice_timings.ndim == 1, "Slice times must be a 1D array"
    n_slices = np.size(slice_timings)
    phase_card = np.zeros((n_scans, n_slices))

    card_peaks_sec = peaks / sample_rate
    for i_slice in range(n_slices):
        # generate slice acquisition timings across all scans
        times_crSlice = t_r * np.arange(n_scans) + slice_timings[i_slice]
        phase_card_crSlice = np.zeros(n_scans)
        for j_scan in range(n_scans):
            previous_card_peaks = np.asarray(
                np.nonzero(card_peaks_sec < times_crSlice[j_scan])
            )
            if np.size(previous_card_peaks) == 0:
                t1 = 0
            else:
                last_peak = previous_card_peaks[0][-1]
                t1 = card_peaks_sec[last_peak]
            next_card_peaks = np.asarray(
                np.nonzero(card_peaks_sec > times_crSlice[j_scan])
            )
            if np.size(next_card_peaks) == 0:
                t2 = n_scans * t_r
            else:
                next_peak = next_card_peaks[0][0]
                t2 = card_peaks_sec[next_peak]
            phase_card_crSlice[j_scan] = (
                2 * np.math.pi * (times_crSlice[j_scan] - t1)
            ) / (t2 - t1)
        phase_card[:, i_slice] = phase_card_crSlice

    return phase_card
