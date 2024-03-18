"""Tests for phys2denoise.metrics.chest_belt."""

import numpy as np
from pytest import mark

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


@mark.xfail
def test_respiratory_phase_smoke():
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


def test_respiratory_pattern_variability_smoke():
    """Basic smoke test for respiratory pattern variability calculation."""
    n_samples = 2000
    resp = np.random.normal(size=n_samples)
    window = 50
    rpv_val = chest_belt.respiratory_pattern_variability(resp, window)
    assert isinstance(rpv_val, float)


def test_env_smoke():
    """Basic smoke test for ENV calculation."""
    n_samples = 2000
    resp = np.random.normal(size=n_samples)
    samplerate = 1 / 0.01
    window = 6
    env_arr = chest_belt.env(resp, samplerate=samplerate, window=window)
    assert env_arr.ndim == 1
    assert env_arr.shape == (n_samples,)


def test_respiratory_variance_smoke():
    """Basic smoke test for respiratory variance calculation."""
    n_samples = 2000
    resp = np.random.normal(size=n_samples)
    samplerate = 1 / 0.01
    window = 6
    rv_arr = chest_belt.respiratory_variance(resp, samplerate=samplerate, window=window)
    assert rv_arr.ndim == 2
    assert rv_arr.shape == (n_samples, 2)
