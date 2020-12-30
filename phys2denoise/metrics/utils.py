import numpy as np
from scipy import signal


def get_butter_filter(fs, lowcut=None, highcut=None, order=5):
    """Calculate the appropriate parameters for a Butterworth filter.

    Parameters
    ----------
    fs : float
        Sampling frequency of data in Hertz.
    lowcut : float or None, optional
        Frequency (in Hertz) under which to remove in the filter.
        If both lowcut and highcut are not None, then a bandpass filter is applied.
        If lowcut is None and highcut is not, then a low-pass filter is applied.
        If highcut is None and lowcut is not, then a high-pass filter is applied.
        Either lowcut, highcut, or both must not be None.
    highcut : float or None, optional
        Frequency (in Hertz) above which to remove in the filter.
        If both lowcut and highcut are not None, then a bandpass filter is applied.
        If lowcut is None and highcut is not, then a low-pass filter is applied.
        If highcut is None and lowcut is not, then a high-pass filter is applied.
        Either lowcut, highcut, or both must not be None.
    order : int, optional
        Nonzero positive integer indicating order of the filter. Default is 1.

    Returns
    -------
    b, a : float
        Parameters from the Butterworth filter.

    Notes
    -----
    From https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    """
    nyq = 0.5 * fs

    if (lowcut is not None) and (highcut is not None):
        low = lowcut / nyq
        high = highcut / nyq
        window = (low, high)
        btype = "bandpass"
    elif lowcut is not None:
        window = lowcut / nyq
        btype = "highpass"
    elif highcut is not None:
        window = highcut / nyq
        btype = "lowpass"
    elif (lowcut is None) and (highcut is None):
        raise ValueError("Either lowcut or highcut must be specified.")

    b, a = signal.butter(order, window, btype=btype)
    return b, a


def butter_bandpass_filter(data, fs, lowcut=None, highcut=None, order=1):
    """Band/low/high-pass filter an array based on cut frequencies.

    Parameters
    ----------
    data : array, shape (n_timepoints,)
        Data to filter.
    fs : float
        Sampling frequency of data in Hertz.
    lowcut : float or None, optional
        Frequency (in Hertz) under which to remove in the filter.
        If both lowcut and highcut are not None, then a bandpass filter is applied.
        If lowcut is None and highcut is not, then a low-pass filter is applied.
        If highcut is None and lowcut is not, then a high-pass filter is applied.
        Either lowcut, highcut, or both must not be None.
    highcut : float or None, optional
        Frequency (in Hertz) above which to remove in the filter.
        If both lowcut and highcut are not None, then a bandpass filter is applied.
        If lowcut is None and highcut is not, then a low-pass filter is applied.
        If highcut is None and lowcut is not, then a high-pass filter is applied.
        Either lowcut, highcut, or both must not be None.
    order : int, optional
        Nonzero positive integer indicating order of the filter. Default is 1.

    Returns
    -------
    y : array, shape (n_timepoints,)
        Filtered data.

    Notes
    -----
    From https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    """
    b, a = get_butter_filter(fs, lowcut, highcut, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def split_complex(complex_signal):
    """Split a complex-valued nifti image into magnitude and phase images.

    From complex-flow
    """
    real = complex_signal.real
    imag = complex_signal.imag
    mag = abs(complex_signal)
    phase = to_phase(real, imag)
    return mag, phase


def to_phase(real, imag):
    """Convert real and imaginary data to phase data.

    Equivalent to cmath.phase.
    https://www.eeweb.com/quizzes/convert-between-real-imaginary-and-magnitude-phase

    From complex-flow
    """
    phase = np.arctan2(imag, real)
    return phase


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Parameters
    ----------
    y : array, shape (n_timepoints,)
        A 1D array with possible NaNs.

    Returns
    -------
    nans : array, shape (n_timepoints,)
        Logical indices of NaNs in y.
    index : function
        A function, with signature indices = index(logical_indices),
        to convert logical indices of NaNs to 'equivalent' indices.

    Examples
    --------
    >>> # linear interpolation of NaNs
    >>> nans, x= nan_helper(y)
    >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])

    Notes
    -----
    From https://stackoverflow.com/a/6520696
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def interp_nans(y):
    """Linearly interpolate across NaNs in a 1D array.

    Parameters
    ----------
    y : array, shape (n_timepoints,)
        A 1D array with possible NaNs.

    Returns
    -------
    y : array, shape (n_timepoints,)
        A 1D array with no NaNs.

    Notes
    -----
    From https://stackoverflow.com/a/6520696
    """
    nans, x = nan_helper(y)
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    return y
