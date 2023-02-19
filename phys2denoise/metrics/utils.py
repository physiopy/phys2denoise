"""Miscellaneous utility functions for metric calculation."""
import logging

import numpy as np

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
    msg = f'The {metric} regressor will be computed using the following parameters:'

    for arg in args:
        msg = f'{msg}\n    {arg} = {args[arg]}'

    msg = f'{msg}\n'

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
            f'Requested buffer size ({buffer}) is longer than input array length '
            f'({len(arr)}). Fixing buffer size to array length.'
        )
        idx = range(arr.shape[0] - len(arr), arr.shape[0])
        pre_mirror = np.take(mirror, idx, axis=0)
        idx = range(len(arr))
        post_mirror = np.take(mirror, idx, axis=0)
    arr_out = np.concatenate((pre_mirror, arr, post_mirror), axis=0)
    return arr_out


def rms_envelope_1d(arr, window=500):
    """Conceptual translation of MATLAB 2017b's envelope(X, x, 'rms') function.

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
        rms = np.sqrt(np.mean(window_arr ** 2))
        rms_env[i] = rms
    rms_env += mean
    return rms_env


def apply_lags(arr1d, lags):
    """Apply delays (lags) to an array.

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
