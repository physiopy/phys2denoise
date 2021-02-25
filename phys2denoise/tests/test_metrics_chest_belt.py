"""Tests for phys2denoise.metrics.chest_belt."""
import numpy as np

from phys2denoise.metrics import chest_belt


def test_rrf_smoke():
    """Basic smoke test for RRF calculation."""
    samplerate = 0.01  # in seconds
    oversampling = 20
    time_length = 20
    onset = 0
    tr = 0.72
    rrf_arr = chest_belt.rrf(
        samplerate,
        oversampling=oversampling,
        time_length=time_length,
        onset=onset,
        tr=tr,
    )
    pred_len = int(np.rint(time_length / (tr / oversampling)))
    assert rrf_arr.ndim == 1
    assert rrf_arr.size == pred_len


def test_respiratory_phase():
    """Basic smoke test for respiratory phase calculation."""
    t_r = 1.0
    n_scans = 200
    sample_rate = 1 / 0.01
    slice_timings = np.linspace(0, t_r, 22)[1:-1]
    n_samples = int(np.rint((n_scans * t_r) * sample_rate))
    resp = np.random.normal(size=n_samples)
    resp_phase = chest_belt.respiratory_phase(
        resp,
        sample_rate=sample_rate,
        slice_timings=slice_timings,
        n_scans=n_scans,
        t_r=t_r,
    )
    assert resp_phase.ndim == 2
    assert resp_phase.shape == (n_scans, slice_timings.size)
