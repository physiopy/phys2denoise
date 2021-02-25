"""These functions compute RETROICOR regressors (Glover et al. 2000)."""

import numpy as np
import matplotlib as mpl

mpl.use("TkAgg")
import matplotlib.pyplot as plt

from .. import references
from ..due import due


def compute_phase_card(card_peaks_timings, slice_timings, n_scans, t_r):
    """Calculate cardiac phase from cardiac peaks.

    Assumes that timing of cardiac events are given in same units
    as slice timings, for example seconds.

    Parameters
    ----------
    peaks : 1D array_like
    slice_timings : 1D array_like
    n_scans : int
    t_r : float

    Returns
    -------
    phase_card : array_like
        Cardiac phase signal.
    """
    n_scans = np.shape(slice_timings)
    phase_card = np.zeros(n_scans)
    for ii in range(n_scans):
        # find previous cardiac peaks
        previous_card_peaks = np.asarray(
            np.nonzero(card_peaks_timings < slice_timings[ii])
        )
        if np.size(previous_card_peaks) == 0:
            t1 = 0
        else:
            last_peak = previous_card_peaks[0][-1]
            t1 = card_peaks_timings[last_peak]

        # find posterior cardiac peaks
        next_card_peaks = np.asarray(np.nonzero(card_peaks_timings > slice_timings[ii]))
        if np.size(next_card_peaks) == 0:
            t2 = n_scans * t_r
        else:
            next_peak = next_card_peaks[0][0]
            t2 = card_peaks_timings[next_peak]

        # compute cardiac phase
        phase_card[ii] = (2 * np.math.pi * (slice_timings[ii] - t1)) / (t2 - t1)

    return phase_card


def compute_phase_resp(resp, sampling_time):
    """Calculate respiratory phase from respiratory signal.

    Parameters
    ----------
    resp
    sampling_time

    Returns
    -------
    phase_resp : array_like
        Respiratory phase signal.
    """
    pass


@due.dcite(references.GLOVER_2000)
def compute_retroicor_regressors(
    physio, t_r, n_scans, slice_timings, n_harmonics, card=False, resp=False
):
    """Compute RETROICOR regressors.

    Parameters
    ----------
    physio : array_like
        1D array, whether cardiac or respiratory signal.
    t_r : float
    n_scans : int
    slice_timings
    n_harmonics : int
    card : bool, optional
    resp : bool, optional

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

    # if respiration, compute histogram and temporal derivative of respiration signal
    if resp:
        # TODO: Replace with numpy.histogram
        resp_hist, resp_hist_bins = plt.hist(physio, bins=100)
        resp_diff = np.diff(physio, n=1)

    # initialize output variables
    retroicor_regressors = []
    phase = np.empty((n_scans, n_slices))

    for i_slice in range(n_slices):
        # Initialize slice timings for current slice
        crslice_timings = t_r * np.arange(n_scans) + slice_timings[i_slice]
        # Compute physiological phases using the timings of physio events (e.g. peaks) slice sampling times
        if card:
            phase[:, i_slice] = compute_phase_card(physio, crslice_timings)
        if resp:
            phase[:, i_slice] = compute_phase_resp(
                resp_diff, resp_hist, resp_hist_bins, crslice_timings
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
