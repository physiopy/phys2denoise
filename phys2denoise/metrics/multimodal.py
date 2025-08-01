"""These functions compute RETROICOR regressors (Glover et al. 2000)."""

from copy import deepcopy as dc

import numpy as np
from peakdet import Physio as pkphysio
from physutils import io, physio

from .. import references
from ..due import due
from .phase import cardiac, respiratory


@due.dcite(references.GLOVER_2000)
def retroicor(
    slice_timings,
    resp_data=None,
    card_data=None,
    resp_fs=None,
    card_fs=None,
    n_harmonics=2,
    base_offset=0,
    nbins="p2d",
    compute_interaction=True,
):
    """Compute RETROICOR regressors.

    Parameters
    ----------
    slice_timings : float or array_like
        Slice times, in seconds.
    resp_data : None, physutils.Physio, or array-like object, optional
        Respiratory data, either a physio object or an array-like object describing the
        recorded respiratory changes *timeseries*
    card_data : None, physutils.Physio, or array-like object, optional
        Cardiac data, either a physio object or an array-like object describing the
        recorded cardiac *peaks' indices* (!!!)
    resp_fs : None or int, optional
        Respiratory data sampling rate, optional if resp_data is a physio object,
        necessary otherwise.
    card_fs : None or int, optional
        Cardiac data sampling rate, optional if card_data is a physio object,
        necessary otherwise.
    n_harmonics : int > 0, optional
        Number of harmonics to compute (for all modalities). Must be > 0. Default is 2.
    base_offset : int or float >=0, optional
        Offset time for the whole neuroimaging data wrt physiological data, default is 0.
        Must be >=0. Negative offsets are currently not supported.
        Note that neuroimaging data offset is normally found in BIDS as a metadata of the
        physiological recording, so the signal must be inverted compared to that field.
    nbins : int or string, optional
        Number of bins to consider in making the histogram or method to estimate such
        number when computing respiratory phase. The default option is "p2d" and
        corresponds to either a quarter of the data amount or 100, whatever is higher.
        Any option supported by `np.histogram` is also supported here.
    compute_interaction : bool, optional
        If respiratory and cardiac data are both given, compute their interactive terms,
        see [2]
        Default is True.

    Notes
    -----
    RETROICOR regressors should be regressed from the imaging data *before*
    any other preprocessing, including slice-timing correction and motion correction,
    and it should be voxel or slice specific.
    Despite this, many people regress one single set of regressors in the GLM - in that
    case, don't use any offset.

    This function does not work like the others due to the issue of having multiple
    possible physio objects in entrance. It will return all physio objects with all
    computed retroicor regressors, so pay attention to repetitions!

    Does not support voxelwise correction yet, thus coefficients are not given.

    References
    ----------
    .. [1] G. H. Glover & T. Q. L. Ress, “Image_based method for retrospective
       correction of physiological motion effects in fMRI: RETROICOR“, Magn. Reson. Med.,
       issue 1, vol. 44, pp. 162-167, 2000.
    .. [2] A.K. Harvey & all, "Brainstem functional magnetic resonance imaging:
       Disentangling signal from physiological noise", JMRI, issue 6, vol. 28, 2008.

    Returns
    -------
    retroicor_regressors : dict
        Retroicor regressors are organised in a dictionary of two dictionaries, one for
        retroicor regressors and one for phases. These dictionaries have one entry per
        slice.
        All entries are lists, always organised: cardiac regressors first, then
        respiratory, then interactions.

        If no physutils.Physio object is given, they will be returned as they are.
    resp_data : physutils.Physio object
        If resp_data is given as physutils.Physio object, returns it with the computed
        retroicor regressors in its metrics metadata.
    card_data : physutils.Physio object
        If card_data is given as physutils.Physio object, returns it with the computed
        retroicor regressors in its metrics metadata.
    """
    if n_harmonics <= 0:
        raise ValueError(
            f"The number of harmonics must be > 0 but {n_harmonics} was given."
        )

    # Parse input data
    return_physio = False

    if resp_data is not None:
        if isinstance(resp_data, physio.Physio):
            # Initialize physio object
            resp_data = physio.check_physio(resp_data, ensure_fs=True, copy=True)
            return_physio = True
        elif isinstance(resp_data, pkphysio):
            # Retrocompatibility with peakdet
            resp_data = io.load_physio(resp_data.data, fs=resp_data.fs)
        elif resp_fs is not None:
            resp_data = physio.Physio(resp_data, fs=resp_fs)
        else:
            raise ValueError(
                """
                resp_data is not a Physio object but resp_fs was not specified.
                To use this function with respiratory data you should either provide a
                Physio object describing the physiological data timeseries, or an
                array-like object *and* the sampling frequency.
                """
            )
    elif card_data is None:
        return None

    if card_data is not None:
        if isinstance(card_data, physio.Physio):
            # Initialize physio object
            card_data = physio.check_physio(card_data, ensure_fs=True, copy=True)
            return_physio = True
        elif isinstance(card_data, pkphysio):
            # Retrocompatibility with peakdet
            peaks = dc(card_data.peaks)
            card_data = io.load_physio(card_data.data, fs=card_data.fs)
            card_data._metadata["peaks"] = peaks
        elif card_fs is not None:
            peaks = dc(card_data)
            card_data = physio.Physio(np.zeros((card_data[-1] + 1)), fs=card_fs)
            card_data._metadata["peaks"] = peaks
        else:
            raise ValueError(
                """
                card_data is not a Physio object but card_fs was not specified.
                To use this function with cardiac data you should either provide a
                Physio object with cardiac data timeseries and peaks, or an
                array-like object describing peaks *and* the sampling frequency.
                """
            )

        if not (hasattr(card_data, "peaks") and np.any(card_data.peaks)):
            raise ValueError(
                """
                Peaks must be a non-empty list for cardiac data.
                Make sure to run peak detection on your cardiac data first,
                using the peakdet module, or other software of your choice.
                """
            )

    slice_timings = np.asarray(slice_timings)
    if slice_timings.squeeze().ndim > 1:
        raise ValueError(
            "The provided slice timings are in a multidimensional array. Please "
            "provide a single 1D array-like data."
        )

    # Initialize output variable
    retroicor_regressors = {}
    phases = {}

    # Compute slice-dependent retroicor regressors.
    for n, slice_time in enumerate(slice_timings):
        phases[n] = {}
        offset = base_offset + slice_time
        # Compute physiological phases using the timings of physio events (e.g. peaks)
        # slice sampling times
        if card_data is not None:
            phases[n]["card"] = cardiac(card_data.peaks, card_data.fs, offset)

        if resp_data is not None:
            phases[n]["resp"] = respiratory(resp_data.data, resp_data.fs, offset, nbins)

            if card_data is not None:
                # Cut to same length. It'll be better later.
                length = np.min((phases[n]["card"].size, phases[n]["resp"].size))
                phases[n]["card"] = phases[n]["card"][:length]
                phases[n]["resp"] = phases[n]["resp"][:length]

        retroicor_regressors[n] = []

        # It's not computationally efficient to loop multiple times per if statement,
        # but organising regressors this way is better for users.
        if card_data is not None:
            for m in n_harmonics:
                retroicor_regressors[n] = retroicor_regressors[n] + [
                    np.cos(m * phases[n]["card"]),
                    np.sin(m * phases[n]["card"]),
                ]

        if resp_data is not None:
            for m in n_harmonics:
                retroicor_regressors[n] = retroicor_regressors[n] + [
                    np.cos(m * phases[n]["resp"]),
                    np.sin(m * phases[n]["resp"]),
                ]

            if compute_interaction and card_data is not None:
                for m in n_harmonics:
                    retroicor_regressors[n] = retroicor_regressors[n] + [
                        np.cos(m * phases[n]["card"]) * np.cos(m * phases[n]["resp"]),
                        np.cos(m * phases[n]["card"]) * np.sin(m * phases[n]["resp"]),
                        np.sin(m * phases[n]["card"]) * np.cos(m * phases[n]["resp"]),
                        np.sin(m * phases[n]["card"]) * np.sin(m * phases[n]["resp"]),
                    ]

    retroicor_regressors = {"retroicor": retroicor_regressors, "phases": phases}

    if return_physio:
        if card_data is not None:
            card_data._computed_metrics["retroicor_regressors"] = retroicor_regressors

        if resp_data is not None:
            resp_data._computed_metrics["retroicor_regressors"] = retroicor_regressors

            if card_data is not None:
                return resp_data, card_data
            else:
                return resp_data
        else:
            return card_data
    else:
        return retroicor_regressors
