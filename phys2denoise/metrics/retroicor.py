"""These functions compute RETROICOR regressors (Glover et al. 2000)."""

import numpy as np
import matplotlib as mpl

mpl.use("TkAgg")
import matplotlib.pyplot as plt

from .. import references
from ..due import due


def compute_phase_card(peaks, slice_timings, n_scans, t_r):
    """Calculate cardiac phase from cardiac peaks.

    Assumes that timing of cardiac events are given in same units
    as slice timings, for example seconds.

    Parameters
    ----------
    peaks : 1D array_like
        Cardiac peak times, in seconds.
    slice_timings : 1D array_like
        Slice times, in seconds.
    n_scans : int
        Number of volumes in the imaging run.
    t_r : float
        Sampling rate of the imaging run, in seconds.

    Returns
    -------
    phase_card : array_like
        Cardiac phase signal.
    """
    n_slices = np.shape(slice_timings)
    phase_card = np.zeros(n_scans)

    for i_slice in range(n_slices):
        # find previous cardiac peaks
        previous_card_peaks = np.asarray(np.nonzero(peaks < slice_timings[i_slice]))
        if np.size(previous_card_peaks) == 0:
            t1 = 0
        else:
            last_peak = previous_card_peaks[0][-1]
            t1 = peaks[last_peak]

        # find posterior cardiac peaks
        next_card_peaks = np.asarray(np.nonzero(peaks > slice_timings[i_slice]))
        if np.size(next_card_peaks) == 0:
            t2 = n_scans * t_r
        else:
            next_peak = next_card_peaks[0][0]
            t2 = peaks[next_peak]

        # compute cardiac phase
        phase_card[i_slice] = (2 * np.math.pi * (slice_timings[i_slice] - t1)) / (
            t2 - t1
        )

    return phase_card


def compute_phase_resp(resp, sample_rate, n_scans, slice_timings, t_r):
    """Calculate respiratory phase from respiratory signal.

    Parameters
    ----------
    resp : 1D array_like
        Respiratory signal.
    sample_rate : float
        Sample rate of physio, in Hertz.
    n_scans
        Number of volumes in the imaging run.
    slice_timings
        Slice times, in seconds.
    t_r
        Sample rate of the imaging run, in seconds.

    Returns
    -------
    phase_resp : array_like
        Respiratory phase signal.
    """
    n_slices = np.shape(slice_timings)
    phase_resp = np.zeros((n_scans, n_slices))

    # generate histogram from respiratory signal
    # TODO: Replace with numpy.histogram
    resp_hist, resp_hist_bins = plt.hist(resp, bins=100)

    # first compute derivative of respiration signal
    resp_diff = np.diff(resp, n=1)

    for i_slice in range(n_slices):
        # generate slice acquisition timings across all scans
        times_crSlice = t_r * np.arange(n_scans) + slice_timings[i_slice]
        phase_resp_crSlice = np.zeros(n_scans)
        for j_scan in range(n_scans):
            iphys = int(
                max([1, round(times_crSlice[j_scan] * sample_rate)])
            )  # closest idx in resp waveform
            iphys = min([iphys, len(resp_diff)])  # cannot be longer than resp_diff
            thisBin = np.argmin(abs(resp[iphys] - resp_hist_bins))
            numerator = np.sum(resp_hist[0:thisBin])
            phase_resp_crSlice[j_scan] = (
                np.math.pi * np.sign(resp_diff[iphys]) * (numerator / len(resp))
            )

        phase_resp[:, i_slice] = phase_resp_crSlice

    return phase_resp


@due.dcite(references.GLOVER_2000)
def retroicor(
    physio,
    sample_rate,
    t_r,
    n_scans,
    slice_timings,
    n_harmonics,
    card=False,
    resp=False,
):
    """Compute RETROICOR regressors.

    Parameters
    ----------
    physio : array_like
        1D array, whether cardiac or respiratory signal.
        If cardiac, the array is a set of peaks in seconds.
        If respiratory, the array is the actual respiratory signal.
    sample_rate : float
        Physio sample rate, in Hertz.
    t_r : float
        Imaging sample rate, in seconds.
    n_scans : int
        Number of volumes in the imaging run.
    slice_timings : array_like
        Slice times, in seconds.
    n_harmonics : int
        ???
    card : bool, optional
        Whether the physio data correspond to cardiac or repiratory signal.
    resp : bool, optional
        Whether the physio data correspond to cardiac or repiratory signal.

    Returns
    -------
    retroicor_regressors : list
    phase : array_like
        2D array of shape (n_scans, n_slices)

    References
    ----------
    *   Glover, G. H., Li, T. Q., & Ress, D. (2000).
        Image‐based method for retrospective correction of physiological
        motion effects in fMRI: RETROICOR.
        Magnetic Resonance in Medicine:
        An Official Journal of the International Society for Magnetic Resonance in Medicine,
        44(1), 162-167.
    """
    n_slices = np.shape(slice_timings)  # number of slices

    # initialize output variables
    retroicor_regressors = []
    phase = np.empty((n_scans, n_slices))

    for i_slice in range(n_slices):
        # Initialize slice timings for current slice
        crslice_timings = t_r * np.arange(n_scans) + slice_timings[i_slice]

        # Compute physiological phases using the timings of physio events (e.g. peaks)
        # slice sampling times
        if card:
            phase[:, i_slice] = compute_phase_card(
                physio,
                crslice_timings,
                n_scans,
                t_r,
            )

        if resp:
            phase[:, i_slice] = compute_phase_resp(
                physio,
                sample_rate,
                n_scans,
                slice_timings,
                t_r,
            )

        # Compute retroicor regressors
        for j_harm in range(n_harmonics):
            retroicor_regressors[i_slice][:, 2 * j_harm] = np.cos(
                (j_harm + 1) * phase[i_slice]
            )
            retroicor_regressors[i_slice][:, 2 * j_harm + 1] = np.sin(
                (j_harm + 1) * phase[i_slice]
            )

    return retroicor_regressors, phase
