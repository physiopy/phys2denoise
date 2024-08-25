"""Tests for phys2denoise.metrics.cardiac."""
import numpy as np
from loguru import logger
from physutils import physio

from phys2denoise.metrics import cardiac


def test_crf_smoke():
    """Basic smoke test for CRF calculation."""
    samplerate = 0.01  # in seconds
    time_length = 20
    onset = 0
    tr = 0.72
    crf_arr = cardiac.crf(
        samplerate, time_length=time_length, onset=onset, inverse=False
    )
    pred_len = np.rint(time_length * 1 / samplerate)
    assert crf_arr.ndim == 1
    assert len(crf_arr) == pred_len


def test_cardiac_phase_smoke():
    """Basic smoke test for cardiac phase calculation."""
    t_r = 1.0
    n_scans = 200
    sample_rate = 1 / 0.01
    slice_timings = np.linspace(0, t_r, 22)[1:-1]
    peaks = np.array([0.534, 0.577, 10.45, 20.66, 50.55, 90.22])
    data = np.zeros(peaks.shape)
    card_phase = cardiac.cardiac_phase(
        data,
        peaks=peaks,
        fs=sample_rate,
        slice_timings=slice_timings,
        n_scans=n_scans,
        t_r=t_r,
    )
    assert card_phase.ndim == 2
    assert card_phase.shape == (n_scans, slice_timings.size)


def test_cardiac_phase_smoke_physio_obj():
    """Basic smoke test for cardiac phase calculation."""
    t_r = 1.0
    n_scans = 200
    sample_rate = 1 / 0.01
    slice_timings = np.linspace(0, t_r, 22)[1:-1]
    peaks = np.array([0.534, 0.577, 10.45, 20.66, 50.55, 90.22])
    data = np.zeros(peaks.shape)
    phys = physio.Physio(data, sample_rate, physio_type="cardiac")
    phys._metadata["peaks"] = peaks

    # Test where the physio object is returned
    phys = cardiac.cardiac_phase(
        phys,
        slice_timings=slice_timings,
        n_scans=n_scans,
        t_r=t_r,
    )

    assert phys.history[0][0] == "phys2denoise.metrics.cardiac.cardiac_phase"
    assert phys.computed_metrics["cardiac_phase"].ndim == 2
    assert phys.computed_metrics["cardiac_phase"].shape == (
        n_scans,
        slice_timings.size,
    )
    assert phys.computed_metrics["cardiac_phase"].args["slice_timings"] is not None
    assert phys.computed_metrics["cardiac_phase"].args["n_scans"] is not None
    assert phys.computed_metrics["cardiac_phase"].args["t_r"] is not None

    # Test where the metric is returned
    card_phase = cardiac.cardiac_phase(
        phys,
        slice_timings=slice_timings,
        n_scans=n_scans,
        t_r=t_r,
        return_physio=False,
    )
    assert card_phase.ndim == 2
    assert card_phase.shape == (n_scans, slice_timings.size)
