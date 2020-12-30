import numpy as np
from scipy import signal

from .utils import butter_bandpass_filter, interp_nans, split_complex


def hilbert_rvt(resp, sampling_freq):
    """Calculate a Hilbert transform-based version of the respiratory volume per unit time metric.

    Parameters
    ----------
    resp : array, shape (n_timepoints,)
        The respiratory trace signal in a 1D array.
    sampling_freq : float
        The sampling frequency, in Hertz.

    Returns
    -------
    rvt : array, shape (n_timepoints,)
        The respiratory volume per unit time metric.

    References
    ----------
    * Harrison, S. J., Bianchi, S., Heinzle, J., Stephan, K. E., Iglesias, S.,
      & Kasper, L. (2020). A Hilbert-based method for processing respiratory timeseries.
      bioRxiv. https://doi.org/10.1101/2020.09.30.321562
    """
    # Remove low frequency drifts (less than 0.01 Hz) from the breathing signal,
    # and remove high-frequency noise above 2.0 Hz.
    # Lowpass filter the data again to more aggressively remove high-frequency noise above 0.75 Hz.
    # Simplified to just bandpass filtering between 0.01-0.75 in one go.
    resp_filt = butter_bandpass_filter(
        resp, fs=sampling_freq, lowcut=0.01, highcut=0.75, order=1
    )

    # Decompose the signal into magnitude and phase components via the Hilbert transform.
    analytic_signal = signal.hilbert(resp_filt)
    magn, phas = split_complex(analytic_signal)

    for i in range(10):
        # Linearly interpolate any periods where the phase time course decreases,
        # using the procedure in Figure 2, to remove any artefactual negative frequencies.
        bad_idx = phas < 0
        phas[bad_idx] = np.nan
        phas = interp_nans(phas)
        # Reconstruct the oscillatory portion of the signal, cos(𝜙(𝑡)),
        # and lowpass filter at 0.75 Hz to remove any resulting artefacts.
        oscill = np.cos(phas)
        oscill = butter_bandpass_filter(oscill, fs=sampling_freq, highcut=0.75)
        phas = np.arccos(oscill)
        # This procedure is repeated 10 times, with the new phase timecourse
        # re-estimated from the filtered oscillatory signal.

    # instantaneous breathing rate is temporal derivative of phase
    # not sure why pi is here but it's in the preprint's formula
    ibr = np.append(0, np.diff(phas)) / (2 * np.pi)

    # respiratory volume is twice signal amplitude
    rv = 2 * magn

    # low-pass filter both(?) at 0.2Hz
    ibr = butter_bandpass_filter(ibr, fs=sampling_freq, highcut=0.75)
    rv = butter_bandpass_filter(rv, fs=sampling_freq, highcut=0.75)

    # rvt is product of ibr and rv
    rvt = ibr * rv

    return rvt
