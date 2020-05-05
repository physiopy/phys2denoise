import numpy as np


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
    arr_with_lags = np.zeros((rv_arr.shape[0], len(lags)))
    for i_lag, lag in enumerate(lags):
        if lag < 0:
            arr_delayed = np.hstack((arr1d[delay:], np.zeros(delay)))
        elif lag > 0:
            arr_delayed = np.hstack((np.zeros(delay), arr1d[delay:]))
        else:
            arr_delayed = arr1d.copy()
        arr_with_lags[:, i_lag] = arr_delayed
    return arr_with_lags
