"""Miscellaneous utility functions for metric calculation."""

import logging

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv
from scipy.interpolate import interp1d
from scipy.stats import zscore

LGR = logging.getLogger(__name__)
LGR.setLevel(logging.INFO)


def print_metric_call(metric, args):
    """
    Log a message to describe how a metric is being called.

    Parameters
    ----------
    metric : function
        Metric function that is being called
    args : dict
        Dictionary containing all arguments that are used to parametrise metric

    Notes
    -----
    Outcome
        An info-level message for the logger.
    """
    msg = f"The {metric} regressor will be computed using the following parameters:"

    for arg in args:
        msg = f"{msg}\n    {arg} = {args[arg]}"

    msg = f"{msg}\n"

    LGR.info(msg)


def mirrorpad_1d(arr, buffer=250):
    """
    Pad both sides of array with flipped values from array of length 'buffer'.

    Parameters
    ----------
    arr
    buffer

    Returns
    -------
    arr_out
    """
    mirror = np.flip(arr, axis=0)
    # If buffer is too long, fix it and issue a warning
    try:
        idx = range(arr.shape[0] - buffer, arr.shape[0])
        pre_mirror = np.take(mirror, idx, axis=0)
        idx = range(0, buffer)
        post_mirror = np.take(mirror, idx, axis=0)
    except IndexError:
        len(arr)
        LGR.warning(
            f"Requested buffer size ({buffer}) is longer than input array length "
            f"({len(arr)}). Fixing buffer size to array length."
        )
        idx = range(arr.shape[0] - len(arr), arr.shape[0])
        pre_mirror = np.take(mirror, idx, axis=0)
        idx = range(len(arr))
        post_mirror = np.take(mirror, idx, axis=0)
    arr_out = np.concatenate((pre_mirror, arr, post_mirror), axis=0)
    return arr_out


def rms_envelope_1d(arr, window=500):
    """
    Conceptual translation of MATLAB 2017b's envelope(X, x, 'rms') function.

    Parameters
    ----------
    arr
    window

    Returns
    -------
    rms_env : numpy.ndarray
        The upper envelope.

    Notes
    -----
    https://www.mathworks.com/help/signal/ref/envelope.html
    """
    assert arr.ndim == 1, "Input data must be 1D"
    assert window % 2 == 0, "Window must be even"
    n_t = arr.shape[0]
    buf = int(window / 2)

    # Pad array at both ends
    arr = np.copy(arr).astype(float)
    mean = np.mean(arr)
    arr -= mean
    arr = mirrorpad_1d(arr, buffer=buf)
    rms_env = np.empty(n_t)
    for i in range(n_t):
        # to match matlab
        start_idx = i + buf
        stop_idx = i + buf + window

        # but this is probably more appropriate
        # start_idx = i + buf - 1
        # stop_idx = i + buf + window
        window_arr = arr[start_idx:stop_idx]
        rms = np.sqrt(np.mean(window_arr**2))
        rms_env[i] = rms
    rms_env += mean
    return rms_env


def apply_lags(arr1d, lags):
    """
    Apply delays (lags) to an array.

    Parameters
    ----------
    arr1d : (X,) :obj:`numpy.ndarray`
        One-dimensional array to apply delays to.
    lags : (Y,) :obj:`tuple` or :obj:`int`
        Delays, in the same units as arr1d, to apply to arr1d. Can be negative,
        zero, or positive integers.

    Returns
    -------
    arr_with_lags : (X, Y) :obj:`numpy.ndarray`
        arr1d shifted according to lags. Each column corresponds to a lag.
    """
    arr_with_lags = np.zeros((arr1d.shape[0], len(lags)))
    for i_lag, lag in enumerate(lags):
        if lag < 0:
            arr_delayed = np.hstack((arr1d[lag:], np.zeros(lag)))
        elif lag > 0:
            arr_delayed = np.hstack((np.zeros(lag), arr1d[lag:]))
        else:
            arr_delayed = arr1d.copy()
        arr_with_lags[:, i_lag] = arr_delayed
    return arr_with_lags


def apply_function_in_sliding_window(array, func, halfwindow, incomplete=True):
    """
    Apply function f in a sliding window view of an array.

    Windows are always considered as centered.
    This function can consider incomplete windows, i.e. those windows
    at the beginning and at the end of an array, so that the length
    of the output is the same as the length of the input. For the same reason,
    it will skip the very last window.

    This is somewhat equivalent to pandas' rolling function set with center=True,
    except for the incomplete windows.

    Parameters
    ----------
    array : list or numpy.ndarray
        Array to apply function in sliding windows to
    func : function
        The bare function to be applied, e.g. np.mean
    halfwindow : int
        Half of the window size to be applied
    incomplete : bool, optional
        If True, return those windows that are smaller, i.e. at the beginning and
        at the end of `array`. If `False`, returns only complete windows.

    Returns
    -------
    numpy.ndarray
        The result of the function on the given array.
    """
    array_out = func(swv(array, halfwindow * 2), axis=1)

    if incomplete:
        for i in reversed(range(halfwindow)):
            array_out = np.append(func(array[: i + halfwindow]), array_out)

        # We're skipping the very last sample to have the same size
        for i in range(-halfwindow + 1, 0):
            array_out = np.append(array_out, func(array[i - halfwindow :]))

    array_out[np.isnan(array_out)] = 0.0

    return array_out


def convolve_and_rescale(array, func, rescale="rescale", pad=False):
    """
    Convolve array by func and rescale the data.

    Parameters
    ----------
    array : list or numpy.ndarray
        Array to be convolved
    func : list or numpy.ndarray
        The function to convolve `array` with
    zscore : bool, optional.
        If True, `array` will be transformed to Zscores before the convolution.
        If False, raw `array` data will be taken to be convolved with the function.
    rescale : "demean_rescale", "rescale", "zscore", "demean", or None, optional
        The rescaling operation used on `array_combined``
    pad : bool, optional
        If True, return a padded non-convolved metric together with the convolved one.
        If False, return both metrics at the lenght of input array.

    Returns
    -------
    array_combined : numpy.ndarray
        One combined array (`array` and `array` convolved with `func`) rescaled or not
    array_combined_padd : numpy.ndarray
        One combined array (`array` and `array` convolved with `func`), padded to the
        convolved data length, rescaled or not.
    """
    # Demeaning before the convolution
    array_dm = array - array.mean(axis=0)
    array_conv = np.convolve(array_dm, func)

    # Stack the array with the convolved array
    if pad:
        endpad = array_conv.shape[0] - array.shape[0]
        endval = array.mean()
        array_combined = np.stack(
            (np.pad(array, (0, endpad), constant_values=endval), array_conv), axis=-1
        )
    else:
        array_combined = np.stack((array, array_conv[: array.shape[0]]), axis=-1)

    # Rescale the combined array
    if rescale == "demean_rescale":
        array_combined = array_combined - array_combined.mean(axis=0)
        array_combined[:, 1] = np.interp(
            array_combined[:, 1],
            (array_combined[:, 1].min(), array_combined[:, 1].max()),
            (array.min(), array.max()),
        )
    elif rescale == "rescale":
        array_combined[:, 1] = np.interp(
            array_combined[:, 1],
            (array_combined[:, 1].min(), array_combined[:, 1].max()),
            (array.min(), array.max()),
        )
    elif rescale == "zscore":
        array_combined = zscore(array_combined, axis=0)
    elif rescale == "demean":
        array_combined = array_combined - array_combined.mean(axis=0)
    else:
        pass

    return array_combined


def export_metric(
    metric,
    sample_rate,
    tr,
    fileprefix,
    ntp=None,
    ext=".1D",
    is_convolved=True,
    has_lags=False,
):
    """
    Export the metric content, both in original sampling rate and resampled at the TR.

    Parameters
    ----------
    metric : list or numpy.ndarray
        Metric to be exported
    sample_rate : int or float
        Original sampling rate of the metric
    tr : int or float
        TR of functional data. Output will be also resampled to this value
    fileprefix : str
        Filename prefix, including path where files should be stored
    ntp : int or None, optional
        Number of timepoints to consider, if None, all will be automatically considered
    ext : str, optional
        Extension of file, default "1D"
    is_convolved : bool, optional.
        If True, `metric` contains convolved version already - default is True
    has_lags : bool, optional.
        If True, `metric` contains lagged versions of itself - default is False
    """
    # Start resampling
    len_tp = metric.shape[0]
    len_newtp = int(np.around(metric.shape[0] * (1 / (sample_rate * tr))))
    len_s = len_tp / sample_rate
    orig_t = np.linspace(0, len_s, len_tp)
    interp_t = np.linspace(0, len_s, len_newtp)
    f = interp1d(orig_t, metric, fill_value="extrapolate", axis=0)

    resampled_metric = f(interp_t)
    if ntp is not None:
        if resampled_metric.shape[-1] > ntp:
            resampled_metric = resampled_metric[:ntp]
        elif resampled_metric.shape[-1] < ntp:
            resampled_metric = np.pad(
                resampled_metric.T, (0, ntp - resampled_metric.shape[-1]), mode="edge"
            ).T

    # Export metrics
    if metric.ndim == 1:
        np.savetxt(f"{fileprefix}_orig{ext}", metric, fmt="%.6f")
        np.savetxt(f"{fileprefix}_resampled{ext}", resampled_metric, fmt="%.6f")
    elif metric.ndim == 2:
        cols = metric.shape[1]
        if cols == 1:
            np.savetxt(f"{fileprefix}_orig{ext}", metric, fmt="%.6f")
            np.savetxt(f"{fileprefix}_resampled{ext}", resampled_metric, fmt="%.6f")
        elif is_convolved:
            np.savetxt(f"{fileprefix}_orig_raw{ext}", metric[:, 0], fmt="%.6f")
            np.savetxt(
                f"{fileprefix}_resampled_raw{ext}", resampled_metric[:, 0], fmt="%.6f"
            )
            np.savetxt(f"{fileprefix}_orig_convolved{ext}", metric[:, 1], fmt="%.6f")
            np.savetxt(
                f"{fileprefix}_resampled_convolved{ext}",
                resampled_metric[:, 1],
                fmt="%.6f",
            )
        elif has_lags:
            for c in range(cols):
                np.savetxt(f"{fileprefix}_orig_lag-{c}{ext}", metric[:, c], fmt="%.6f")
                np.savetxt(
                    f"{fileprefix}_resampled_lag-{c}{ext}",
                    resampled_metric[:, c],
                    fmt="%.6f",
                )

    return fileprefix
