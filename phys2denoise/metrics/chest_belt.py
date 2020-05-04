"""Denoising metrics for chest belt recordings
"""


def rpv(belt_ts):
    """Respiratory pattern variability

    1. Z-score respiratory belt signal
    2. Calculate upper envelope
    3. Calculate standard deviation of envelope
    """
    pass


def env(belt_ts, samplerate, out_samplerate, window=10):
    """Respiratory pattern variability calculated across a sliding window

    Across a sliding window, do the following:
    1. Z-score respiratory belt signal
    2. Calculate upper envelope
    3. Calculate standard deviation of envelope
    """
    pass


def rv(belt_ts, samplerate, out_samplerate, window=10):
    """Respiratory variance

    Parameters
    ----------
    belt_ts : :obj:`numpy.ndarray`
        A 1D array with the respiratory belt time series.
    samplerate : :obj:`float`
        Sampling rate for belt_ts, in Hertz.
    out_samplerate : :obj:`float`
        Sampling rate for the output time series. Corresponds to TR in fMRI
        data.
    window : :obj:`int`, optional
        Size of the sliding window. Default is 10.
    """
    pass


def rvt(belt_ts, samplerate, out_samplerate, window=10):
    """Respiratory volume-per-time
    """
    pass


def rrf(samplerate):
    """Respiratory response function
    """
    pass


def convolve_rrf(arr, samplerate):
    # Get respiratory response function (RRF)
    rrf_arr = rrf(samplerate=samplerate)
    # Convolve RV with RRF
    arr_x_rrf = convolve(arr, rrf_arr)
