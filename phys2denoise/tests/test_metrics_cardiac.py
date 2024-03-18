"""Tests for phys2denoise.metrics.cardiac."""

import numpy as np

from phys2denoise.metrics import cardiac


def test_crf_smoke():
    """Basic smoke test for CRF calculation."""
    samplerate = 0.01  # in seconds
    oversampling = 20
    time_length = 20
    onset = 0
    tr = 0.72
    crf_arr = cardiac.crf(
        samplerate,
        oversampling=oversampling,
        time_length=time_length,
        onset=onset,
        tr=tr,
    )
    pred_len = np.rint(time_length / (tr / oversampling))
    assert crf_arr.ndim == 1
    assert crf_arr.shape == pred_len


def test_cardiac_phase_smoke():
    """Basic smoke test for cardiac phase calculation."""
    t_r = 1.0
    n_scans = 200
    sample_rate = 1 / 0.01
    slice_timings = np.linspace(0, t_r, 22)[1:-1]
    peaks = np.array([0.534, 0.577, 10.45, 20.66, 50.55, 90.22])
    card_phase = cardiac.cardiac_phase(
        peaks,
        sample_rate=sample_rate,
        slice_timings=slice_timings,
        n_scans=n_scans,
        t_r=t_r,
    )
    assert card_phase.ndim == 2
    assert card_phase.shape == (n_scans, slice_timings.size)
