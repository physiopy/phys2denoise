"""These functions compute RETROICOR regressors (Glover et al. 2000)."""

import numpy as np

from .. import references
from ..due import due
from .cardiac import cardiac_phase
from .chest_belt import respiratory_phase


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

    Notes
    -----
    RETROICOR regressors should be regressed from the imaging data *before*
    any other preprocessing, including slice-timing correction and motion correction.

    References
    ----------
    .. [1] G. H. Glover & T. Q. L. Ress, “Image_based method for retrospective
       correction of physiological motion effects in fMRI: RETROICOR“, Magn. Reson. Med.,
       issue 1, vol. 44, pp. 162-167, 2000.
    """
    n_slices = np.shape(slice_timings)  # number of slices

    # initialize output variables
    retroicor_regressors = []
    phase = np.empty((n_scans, n_slices))

    for i_slice in range(n_slices):
        retroicor_regressors.append(np.empty((n_scans, 2 * n_harmonics)))

        # Initialize slice timings for current slice
        crslice_timings = t_r * np.arange(n_scans) + slice_timings[i_slice]

        # Compute physiological phases using the timings of physio events (e.g. peaks)
        # slice sampling times
        if card:
            phase[:, i_slice] = cardiac_phase(
                physio,
                crslice_timings,
                n_scans,
                t_r,
            )

        if resp:
            phase[:, i_slice] = respiratory_phase(
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
