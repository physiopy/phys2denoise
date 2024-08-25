"""Tests for phys2denoise.tasks"""
import numpy as np
import pydra
from loguru import logger
from physutils import physio
from pydra import Workflow

from phys2denoise.tasks import compute_metrics, export_metrics


def test_compute_metrics(fake_physio):
    """Test the computation of metrics."""
    t_r = 1.0
    n_scans = 200
    slice_timings = np.linspace(0, t_r, 22)[1:-1]

    task2 = compute_metrics(
        phys=fake_physio,
        metrics=["respiratory_variance", "respiratory_variance_time", "cardiac_phase"],
        args={
            "cardiac_phase": {
                "slice_timings": slice_timings,
                "n_scans": n_scans,
                "t_r": t_r,
            },
        },
    )
    assert task2.inputs.metrics == [
        "respiratory_variance",
        "respiratory_variance_time",
        "cardiac_phase",
    ]
    assert task2.inputs.phys == fake_physio
    assert task2.inputs.args == {
        "cardiac_phase": {
            "slice_timings": slice_timings,
            "n_scans": n_scans,
            "t_r": t_r,
        },
    }
    task2()
    new_physio = task2.result().output.out

    assert new_physio.computed_metrics["respiratory_variance"] is not None
    assert new_physio.computed_metrics["respiratory_variance_time"] is not None
    assert new_physio.computed_metrics["cardiac_phase"] is not None

    assert new_physio.computed_metrics["respiratory_variance"].shape == (
        len(new_physio.data),
        2,
    )
    assert new_physio.computed_metrics["respiratory_variance_time"].shape == (
        len(new_physio.data),
        4,
    )
    assert new_physio.computed_metrics["cardiac_phase"].shape == (
        n_scans,
        slice_timings.size,
    )

    assert new_physio.computed_metrics["respiratory_variance"].args["window"] == 6
    assert new_physio.computed_metrics["respiratory_variance_time"].args["lags"] == (
        0,
        4,
        8,
        12,
    )
    assert (
        new_physio.computed_metrics["cardiac_phase"].args["slice_timings"] is not None
    )
    assert new_physio.computed_metrics["cardiac_phase"].args["n_scans"] is not None
    assert new_physio.computed_metrics["cardiac_phase"].args["t_r"] is not None
