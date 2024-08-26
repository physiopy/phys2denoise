"""Tests for phys2denoise.tasks"""
import os

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


def test_export_metrics(fake_physio_with_metrics):
    """Test the export of metrics."""
    file_dir = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(file_dir, "test_output_data/standalone")
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    tr = 1.0

    task = export_metrics(
        phys=fake_physio_with_metrics,
        metrics=["respiratory_variance", "respiratory_variance_time"],
        outdir=outdir,
        tr=tr,
    )
    assert task.inputs.metrics == ["respiratory_variance", "respiratory_variance_time"]
    assert task.inputs.phys == fake_physio_with_metrics
    assert task.inputs.outdir == outdir
    assert task.inputs.tr == tr
    task()

    # Check if the output directory was created
    assert os.path.exists(outdir)

    # Check if the files were created
    assert os.path.exists(
        os.path.join(outdir, "respiratory_variance_orig_convolved.1D")
    )
    assert os.path.exists(os.path.join(outdir, "respiratory_variance_orig_raw.1D"))
    assert os.path.exists(
        os.path.join(outdir, "respiratory_variance_resampled_convolved.1D")
    )
    assert os.path.exists(os.path.join(outdir, "respiratory_variance_resampled_raw.1D"))
    assert os.path.exists(
        os.path.join(outdir, "respiratory_variance_time_orig_convolved.1D")
    )
    assert os.path.exists(os.path.join(outdir, "respiratory_variance_time_orig_raw.1D"))
    assert os.path.exists(
        os.path.join(outdir, "respiratory_variance_time_resampled_convolved.1D")
    )
    assert os.path.exists(
        os.path.join(outdir, "respiratory_variance_time_resampled_raw.1D")
    )
