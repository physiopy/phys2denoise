"""These functions compute RETROICOR regressors (Glover et al. 2000)."""

import numpy as np
from physutils import io, physio

from .. import references
from ..due import due
from .cardiac import cardiac_phase
from .chest_belt import respiratory_phase
from .utils import return_physio_or_metric


@due.dcite(references.GLOVER_2000)
@return_physio_or_metric()
@physio.make_operation()
def retroicor(
    data,
    t_r,
    n_scans,
    slice_timings,
    n_harmonics,
    physio_type=None,
    fs=None,
    cardiac_peaks=None,
    **kwargs,
):
    """Compute RETROICOR regressors.

    Parameters
    ----------
    data : physutils.Physio, np.ndarray, or array-like object
        Object containing the timeseries of the recorded respiratory or cardiac signal
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
    if isinstance(data, physio.Physio):
        # Initialize physio object
        data = physio.check_physio(data, ensure_fs=True, copy=True)
        if data.physio_type is None and physio_type is not None:
            data._physio_type = physio_type
        elif data.physio_type is None and physio_type is None:
            raise ValueError(
                """
                Since the provided Physio object does not specify a `physio_type`,
                this function's `physio_type` parameter must be specified as a
                value from {'cardiac', 'respiratory'}
                """
            )

    elif fs is not None and physio_type is not None:
        data = io.load_physio(data, fs=fs)
        data._physio_type = physio_type
        if data.physio_type == "cardiac":
            data._metadata["peaks"] = cardiac_peaks
    else:
        raise ValueError(
            """
            To use this function you should either provide a Physio object
            with existing peaks metadata if it describes a cardiac signal
            (e.g. using the peakdet module), or
            by providing the physiological data timeseries, the sampling frequency,
            the physio_type and the peak indices separately.
            """
        )
    if not data.peaks and data.physio_type == "cardiac":
        raise ValueError(
            """
            Peaks must be a non-empty list for cardiac data.
            Make sure to run peak detection on your cardiac data first,
            using the peakdet module, or other software of your choice.
            """
        )

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
        if data.physio_type == "cardiac":
            phase[:, i_slice] = cardiac_phase(
                data,
                crslice_timings,
                n_scans,
                t_r,
            )

        if data.physio_type == "respiratory":
            phase[:, i_slice] = respiratory_phase(
                data,
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

    data._computed_metrics["retroicor_regressors"] = dict(
        metrics=retroicor_regressors, phase=phase
    )
    retroicor_regressors = dict(metrics=retroicor_regressors, phase=phase)

    return data, retroicor_regressors
