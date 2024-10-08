"""Denoising metrics for cardio recordings."""

import numpy as np
from loguru import logger
from physutils import io, physio

from .. import references
from ..due import due
from .responses import crf
from .utils import apply_function_in_sliding_window as afsw
from .utils import convolve_and_rescale, return_physio_or_metric


def _cardiac_metrics(
    data,
    metric,
    peaks=None,
    fs=None,
    window=6,
    central_measure="mean",
    **kwargs,
):
    """
    Compute cardiac metrics.

    Computes the average heart beats interval (HBI)
    or the average heart rate variability (HRV) in a sliding window.

    We refer to HEART RATE (variability), however note that if using a PPG
    recorded signal, it is more accurate to talk about PULSE RATE (variability).
    See [3]_ for the differences and similarities between the two measures.

    Parameters
    ----------
    data : physutils.Physio, np.ndarray, or array-like object
        Object containing the timeseries of the recorded cardiac signal
    metrics : "hbi", "hr", "hrv", string
        Cardiac metric(s) to calculate.
    fs : float, optional if data is a physutils.Physio
        Sampling rate of `data` in Hz
    peaks : :obj:`numpy.ndarray`, optional if data is a physutils.Physio
        Indices of peaks in `data`
    window : float, optional
        Size of the sliding window, in seconds.
        Default is 6.
    central_measure : "mean","average", "avg", "median", "mdn", "stdev", "std", string, optional
        Measure of the center used (mean or median).
        Default is "mean".

    Returns
    -------
    card_met : 2D numpy.ndarray
        Heart Beats Interval or Heart Rate Variability timeseries.
        The first column is the raw metric, in seconds if HBI, in Hertz if HRV.
        The second column is the metric convolved with the CRF, cut to the length
        of the raw metric.

    Notes
    -----
    Heart beats interval (HBI) definition is taken from [1]_, and consists of the
    average of the time interval between two heart beats within a 6-seconds window.
    This metric should be convolved with an inverse of the cardiac response function
    before being included in a GLM.

    Heart rate variability (HRV) is taken from [2]_, and computed as the amounts of
    beats per minute.
    However, operationally, it is the average of the inverse of the time interval
    between two heart beats.
    This metric should be convolved with the cardiac response function
    before being included in a GLM.

    IMPORTANT : Here both metrics' unit of measure have meaning, since they are based
    on seconds/hertz. Hence, zscoring might remove important quantifiable information.

    References
    ----------
    .. [1] J. E. Chen et al., "Resting-state "physiological networks"", Neuroimage,
        vol. 213, pp. 116707, 2020.
    .. [2] C. Chang, J. P. Cunningham, & G. H. Glover, "Influence of heart rate on the
        BOLD signal: The cardiac response function", NeuroImage, vol. 44, 2009
    .. [3] N. Pinheiro et al., "Can PPG be used for HRV analysis?," 2016 38th
    Annual International Conference of the IEEE Engineering in Medicine and
    Biology Society (EMBC), doi: 10.1109/EMBC.2016.7591347.
    """
    if isinstance(data, physio.Physio):
        # Initialize physio object
        data = physio.check_physio(data, ensure_fs=True, copy=True)
    elif fs is not None and peaks is not None:
        data = io.load_physio(data, fs=fs)
        data._metadata["peaks"] = peaks
    else:
        raise ValueError(
            """
            To use this function you should either provide a Physio object
            with existing peaks metadata (e.g. using the peakdet module), or
            by providing the physiological data timeseries, the sampling frequency,
            and the peak indices separately.
            """
        )
    if data.peaks.size == 0:
        raise ValueError(
            """
            Peaks must be a non-empty list.
            Make sure to run peak detection on your physiological data first,
            using the peakdet module, or other software of your choice.
            """
        )

    # Convert window to samples, but halves it.
    halfwindow_samples = int(round(window * data.fs / 2))

    if central_measure in ["mean", "average", "avg"]:
        central_measure_operator = np.mean
    elif central_measure in ["median", "mdn"]:
        central_measure_operator = np.median
    elif central_measure in ["stdev", "std"]:
        central_measure_operator = np.std
    else:
        raise NotImplementedError(
            f" {central_measure} is not a supported metric of centrality."
        )

    idx_arr = np.arange(len(data))
    idx_min = afsw(idx_arr, np.min, halfwindow_samples)
    idx_max = afsw(idx_arr, np.max, halfwindow_samples)

    card_met = np.empty_like(data)
    for n, i in enumerate(idx_min):
        diff = (
            np.diff(
                data.peaks[np.logical_and(data.peaks >= i, data.peaks <= idx_max[n])]
            )
            / data.fs
        )
        if metric == "hbi":
            card_met[n] = central_measure_operator(diff) if diff.size > 0 else 0
        elif metric == "hr":
            card_met[n] = central_measure_operator(1 / diff) if diff.size > 0 else 0
        elif metric == "hrv":
            central_measure_operator = np.std
            card_met[n] = central_measure_operator(1 / diff) if diff.size > 0 else 0
        else:
            raise NotImplementedError(
                f"{metric} is not a supported value for requested cardiac metrics."
            )

    card_met[np.isnan(card_met)] = 0.0

    # Convolve with crf and rescale
    card_met = convolve_and_rescale(card_met, crf(data.fs), rescale="rescale")

    return data, card_met


@due.dcite(references.CHANG_CUNNINGHAM_GLOVER_2009)
@return_physio_or_metric()
@physio.make_operation()
def heart_rate(data, peaks=None, fs=None, window=6, central_measure="mean", **kwargs):
    """
    Compute average heart rate (HR) in a sliding window.

    We call this function HEART RATE, however note that if using a PPG
    recorded signal, it is more accurate to talk about PULSE RATE.
    See [2]_ for the differences and similarities between the two measures.

    Parameters
    ----------
    data : physutils.Physio, np.ndarray, or array-like object
        Object containing the timeseries of the recorded cardiac signal
    fs : float, optional if data is a physutils.Physio
        Sampling rate of `data` in Hz
    peaks : :obj:`numpy.ndarray`, optional if data is a physutils.Physio
        Indices of peaks in `data`
    window : float, optional
        Size of the sliding window, in seconds.
        Default is 6.
    central_measure : "mean","average", "avg", "median", "mdn",  string, optional
        Measure of the center used (mean or median).
        Default is "mean".
    Returns
    -------
    card_met : 2D numpy.ndarray
        Heart Beats Interval or Heart Rate Variability timeseries.
        The first column is the raw metric, in Hertz.
        The second column is the metric convolved with the CRF, cut to the length
        of the raw metric.

    Notes
    -----
    Heart rate (HR) is taken from [1]_, and computed as the amounts of
    beats per minute.
    However, operationally, it is the average of the inverse of the time interval
    between two heart beats.
    This metric should be convolved with the cardiac response function
    before being included in a GLM.

    IMPORTANT : The unit of measure has a meaning, since they it's based on Hertz.
    Hence, zscoring might remove important quantifiable information.

    See `_cardiac_metrics` for full implementation.

    References
    ----------
    .. [1] C. Chang, J. P. Cunningham, & G. H. Glover, "Influence of heart rate on the
        BOLD signal: The cardiac response function", NeuroImage, vol. 44, 2009
    .. [2] N. Pinheiro et al., "Can PPG be used for HRV analysis?," 2016 38th
    Annual International Conference of the IEEE Engineering in Medicine and
    Biology Society (EMBC), doi: 10.1109/EMBC.2016.7591347.
    """
    data, hr = _cardiac_metrics(
        data,
        metric="hr",
        peaks=peaks,
        fs=fs,
        window=window,
        central_measure=central_measure,
        **kwargs,
    )

    return data, hr


@due.dcite(references.PINHERO_ET_AL_2016)
@return_physio_or_metric()
@physio.make_operation()
def heart_rate_variability(
    data, peaks=None, fs=None, window=6, central_measure="mean", **kwargs
):
    """
    Compute average heart rate variability (HRV) in a sliding window.

    We call this function HEART RATE variability, however note that if using a PPG
    recorded signal, it is more accurate to talk about PULSE RATE variability.
    See [1]_ for the differences and similarities between the two measures.

    Parameters
    ----------
    data : physutils.Physio, np.ndarray, or array-like object
        Object containing the timeseries of the recorded cardiac signal
    fs : float, optional if data is a physutils.Physio
        Sampling rate of `data` in Hz
    peaks : :obj:`numpy.ndarray`, optional if data is a physutils.Physio
        Indices of peaks in `data`
    window : float, optional
        Size of the sliding window, in seconds.
        Default is 6.
    central_measure : "mean","average", "avg", "median", "mdn",  string, optional
        Measure of the center used (mean or median).
        Default is "mean".
    Returns
    -------
    card_met : 2D numpy.ndarray
        Heart Beats Interval or Heart Rate Variability timeseries.
        The first column is the raw metric, in Hertz.
        The second column is the metric convolved with the CRF, cut to the length
        of the raw metric.

    Notes
    -----
    Heart rate variability (HRV) is taken from [1]_, and computed as the amounts of
    beats per minute.
    However, operationally, it is the average of the inverse of the time interval
    between two heart beats.
    This metric should be convolved with the cardiac response function
    before being included in a GLM.

    IMPORTANT : The unit of measure has a meaning, since they it's based on Hertz.
    Hence, zscoring might remove important quantifiable information.

    See `_cardiac_metrics` for full implementation.

    References
    ----------
    .. [1] N. Pinheiro et al., "Can PPG be used for HRV analysis?," 2016 38th
    Annual International Conference of the IEEE Engineering in Medicine and
    Biology Society (EMBC), doi: 10.1109/EMBC.2016.7591347.
    """
    data, hrv = _cardiac_metrics(
        data,
        metric="hrv",
        peaks=peaks,
        fs=fs,
        window=window,
        central_measure=central_measure,
        **kwargs,
    )

    return data, hrv


@due.dcite(references.CHEN_2020)
@return_physio_or_metric()
@physio.make_operation()
def heart_beat_interval(
    data, peaks=None, fs=None, window=6, central_measure="mean", **kwargs
):
    """
    Compute average heart beat interval (HBI) in a sliding window.

    Parameters
    ----------
    data : physutils.Physio, np.ndarray, or array-like object
        Object containing the timeseries of the recorded cardiac signal
    fs : float, optional if data is a physutils.Physio
        Sampling rate of `data` in Hz
    peaks : :obj:`numpy.ndarray`, optional if data is a physutils.Physio
        Indices of peaks in `data`
    window : float, optional
        Size of the sliding window, in seconds.
        Default is 6.
    central_measure : "mean","average", "avg", "median", "mdn",  string, optional
        Measure of the center used (mean or median).
        Default is "mean".
    Returns
    -------
    card_met : 2D numpy.ndarray
        Heart Beats Interval or Heart Rate Variability timeseries.
        The first column is the raw metric, in seconds.
        The second column is the metric convolved with the CRF, cut to the length
        of the raw metric.

    Notes
    -----
    Heart beats interval (HBI) definition is taken from [1]_, and consists of the
    average of the time interval between two heart beats within a 6-seconds window.
    This metric should be convolved with an inverse of the cardiac response function
    before being included in a GLM.

    IMPORTANT : The unit of measure has meaning, since it is based on seconds.
    Hence, zscoring might remove important quantifiable information.

    See `_cardiac_metrics` for full implementation.

    References
    ----------
    .. [1] J. E. Chen et al., "Resting-state "physiological networks"", Neuroimage,
        vol. 213, pp. 116707, 2020.
    """
    data, hbi = _cardiac_metrics(
        data,
        metric="hbi",
        peaks=peaks,
        fs=fs,
        window=window,
        central_measure=central_measure,
        **kwargs,
    )

    return data, hbi


@return_physio_or_metric()
@physio.make_operation()
def cardiac_phase(data, slice_timings, n_scans, t_r, peaks=None, fs=None, **kwargs):
    """Calculate cardiac phase from cardiac peaks.

    Assumes that timing of cardiac events are given in same units
    as slice timings, for example seconds.

    Parameters
    ----------
    data : physutils.Physio, np.ndarray, or array-like object
        Object containing the timeseries of the recorded cardiac signal
    slice_timings : 1D array_like
        Slice times, in seconds.
    n_scans : int
        Number of volumes in the imaging run.
    t_r : float
        Sampling rate of the imaging run, in seconds.
    fs : float, optional if data is a physutils.Physio
        Sampling rate of `data` in Hz
    peaks : :obj:`numpy.ndarray`, optional if data is a physutils.Physio
        Indices of peaks in `data`

    Returns
    -------
    phase_card : array_like
        Cardiac phase signal, of shape (n_scans,)
    """
    if isinstance(data, physio.Physio):
        # Initialize physio object
        data = physio.check_physio(data, ensure_fs=True, copy=True)
    elif fs is not None and peaks is not None:
        data = io.load_physio(data, fs=fs)
        data._metadata["peaks"] = peaks
    else:
        raise ValueError(
            """
            To use this function you should either provide a Physio object
            with existing peaks metadata (e.g. using the peakdet module), or
            by providing the physiological data timeseries, the sampling frequency,
            and the peak indices separately.
            """
        )
    if data.peaks.size == 0:
        raise ValueError(
            """
            Peaks must be a non-empty list.
            Make sure to run peak detection on your physiological data first,
            using the peakdet module, or other software of your choice.
            """
        )

    assert slice_timings.ndim == 1, "Slice times must be a 1D array"
    n_slices = np.size(slice_timings)
    phase_card = np.zeros((n_scans, n_slices))

    card_peaks_sec = data.peaks / data.fs
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

    return data, phase_card
