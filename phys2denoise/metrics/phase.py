"""Denoising metrics for cardio recordings."""

import numpy as np
from loguru import logger
from scipy.interpolation import interp1d


def cardiac(peaks, fs, offset=0):
    """Calculate cardiac phase from cardiac peaks.

    Assumes that timing of cardiac events are given in same units
    as slice timings, for example seconds.

    This function will always return just the metric, never a physio object with the
    metric inside.

    Parameters
    ----------
    peaks : 1D array_like
        Indices of peaks in a cardiac array
    fs : float
        Sampling rate of physiological data in Hz
    offset : int or float >=0, optional
        Offset time for the neuroimaging data. This includes slice timing (normally
        positive) and neuroimaging data offset wrt physiological data (normally positive).
        Must be >=0. Negative offsets are currently not supported.
        Note that neuroimaging data offset is normally found in BIDS as a metadata of the
        physiological recording, so the signal must be inverted compared to that field.

    Returns
    -------
    phase_card : array_like
        Cardiac phase signal, sampled at the physiological signal sampling rate and of
        length = peaks[-1]

    Note
    ----
    This function will always return just the metric, never a physio object with the
    metric inside, unlike every other metric function in this module.

    This function won't fail, but you will not be able to export regressors, if the
    physiological data collected ends before the neuroimaging data.

    `time1` and `time2` refer to the original formula from [1]:
        phi(t) = 2*pi*(t-t1)/(t2-t1)
    They represent the beat right before the reference timepoint and the one after that respectively.

    Raises
    ------
    ValueError
        If the offset is negative.

    References
    ----------
    .. [1] G. H. Glover & T. Q. L. Ress, “Image_based method for retrospective
       correction of physiological motion effects in fMRI: RETROICOR“, Magn. Reson. Med.,
       issue 1, vol. 44, pp. 162-167, 2000.
    """
    if offset < 0:
        raise ValueError("Negative offsets are not supported yet.")

    # Add a beat before and after the current beats
    avg_peak_dist = int(np.diff(peaks).mean().round())
    peaks = np.append(
        np.append(peaks[0] - avg_peak_dist, peaks),
        [peaks[-1] + avg_peak_dist, peaks[-1] + 2 * avg_peak_dist],
    )

    # Pre-append additional peaks until we get to <0
    while peaks[0] >= 0:
        peaks = np.append((peaks[0] - avg_peak_dist), peaks)

    # Transform in seconds and offset
    peaks_sec = peaks / fs - offset

    # Create timeline reference for neuroimaging on the peaks time, adding offset,
    # accounting for the two extra peaks added above.
    time = np.arange(peaks[-3] + 1) / fs

    time1 = interp1d(peaks_sec, peaks_sec, kind="previous", assume_sorted=True)(time)
    time2 = interp1d(peaks_sec, peaks_sec, kind="next", assume_sorted=True)(time)

    return 2 * np.pi * ((time[1:] - time1[:-1]) / (time2[1:] - time1[:-1]))


def respiratory(data, fs, offset=0, nbins="p2d"):
    """Calculate respiratory phase from respiratory signal.

    Parameters
    ----------
    data : array-like object
        Recorded respiratory signal's timeseries
    fs : float
        Sampling rate of physiological data in Hz
    offset : int or float >=0, optional
        Offset time for the neuroimaging data, default is 0. This includes slice timing
        (normally positive) and neuroimaging data offset wrt physiological data
        (normally positive).
        Must be >=0. Negative offsets are currently not supported.
        Note that neuroimaging data offset is normally found in BIDS as a metadata of the
        physiological recording, so the signal must be inverted compared to that field.
    nbins : int or string, optional
        Number of bins to consider in making the histogram or method to estimate such
        number. The default option is "p2d" and corresponds to either a quarter of the
        data amount or 100, whatever is higher.
        Any option supported by `np.histogram` is also supported here.

    Returns
    -------
    phase_resp : array_like
        Respiratory phase signal, sampled at the physiological signal sampling rate and
        of length = data.size

    Note
    ----
    This function will always return just the metric, never a physio object with the
    metric inside, unlike every other metric function in this module.

    This function won't fail, but you will not be able to export regressors, if the
    physiological data collected ends before the neuroimaging data.

    This is an adapted and vectorized version of the original formula from [1].
    The original formula has basically three elements for each timepoint:
    - pi
    - the area under the curve (AUC) left of any bin that timepoint is in,
      divided by the total AUC
    - the sign of the (discrete) derivative of the signal at that timepoint

    Here it is implemented as:
    - pi (duh)
    - the cumulative sum of the counts up to the bin of the timepoint, in a histogram
      normalised by the amount of data entries (so that total AUC, i.e. cumsum, = 1)
    - the sign of the one-hop difference of the signal.

    It is, however, exactly the same thing (beside decimal point error).

    Also, while the histogram is created on the original data, the search happens with
    the offsetted data (if offset is specified). This is so that all slice time shifts
    refer to the same histogram.

    Raises
    ------
    ValueError
        If offset is < 0

    References
    ----------
    .. [1] G. H. Glover & T. Q. L. Ress, “Image_based method for retrospective
       correction of physiological motion effects in fMRI: RETROICOR“, Magn. Reson. Med.,
       issue 1, vol. 44, pp. 162-167, 2000.
    """
    if offset < 0:
        raise ValueError("Negative offsets are not supported yet.")
    elif offset > 0:
        # Interpolate data in neuroimaging's offsetted time.
        time = np.arange(data.size)
        data_offsetted = interp1d(
            time, data, kind="linear", fill_value="extrapolate", assume_sorted=True
        )(time + offset)
    else:
        # Skip interpolation
        data_offsetted = data

    nbins = max(100, int(data.size / 4)) if nbins == "p2d" else nbins
    counts, binedge = np.histogram(data, bins=nbins)
    # Normalize counts by total to avoid denominator computation later
    counts = counts / counts.sum()
    # For each element, find its bin
    bincenter = (binedge[:-1] + binedge[1:]) / 2
    idx = np.searchsorted(bincenter, data_offsetted, side="left")
    # Cumulative sum computes AUC left of any bin
    cumsum = np.cumsum(counts)
    # use the bin index to find the AUC left of it.
    raw_phase = np.where(idx > 0, cumsum[idx - 1], 0)

    sign = np.sign(np.diff(data))
    sign = np.append(sign, sign[-1])

    return np.pi * raw_phase * sign
