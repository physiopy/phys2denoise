"""Tests for phys2denoise.tasks and their integration."""
import logging
import os

import nest_asyncio
import numpy as np
from loguru import logger
from physutils import physio
from physutils.tasks import transform_to_physio
from pydra import Submitter, Workflow

import phys2denoise.tasks as tasks

from .utils import create_fake_phys

nest_asyncio.apply()

LGR = logging.getLogger(__name__)
LGR.setLevel(logging.DEBUG)


def test_integration(fake_phys):
    """Test the integration of phys2denoise tasks."""
    file_dir = os.path.dirname(os.path.abspath(__file__))
    export_dir = os.path.join(file_dir, "test_output_data/integration")
    create_fake_phys()

    physio_file = os.path.abspath("phys2denoise/tests/data/fake_phys.phys")

    wf = Workflow(
        name="metrics_wf",
        input_spec=["phys", "fs"],
        phys=physio_file,
    )
    wf.add(
        transform_to_physio(
            name="transform_to_physio",
            input_file=wf.lzin.phys,
            fs=wf.lzin.fs,
            mode="physio",
        )
    )
    wf.add(
        tasks.compute_metrics(
            name="compute_metrics",
            phys=wf.transform_to_physio.lzout.out,
            metrics=[
                "respiratory_variance",
                "respiratory_variance_time",
                "cardiac_phase",
            ],
            args={
                "cardiac_phase": {
                    "slice_timings": np.linspace(0, 1, 22)[1:-1],
                    "n_scans": 200,
                    "t_r": 1.0,
                },
            },
        )
    )
    wf.add(
        tasks.export_metrics(
            name="export_metrics",
            phys=wf.compute_metrics.lzout.out,
            metrics="all",
            outdir=export_dir,
            tr=1.0,
        )
    )
    wf.set_output([("result", wf.compute_metrics.lzout.out)])

    with Submitter(plugin="cf") as sub:
        sub(wf)
    wf()

    output_physio = wf.result().output.result
    LGR.debug(f"Output physio: {output_physio}")

    # Physio object assertions
    assert output_physio.computed_metrics["respiratory_variance"] is not None
    assert output_physio.computed_metrics["respiratory_variance_time"] is not None
    assert output_physio.computed_metrics["cardiac_phase"] is not None

    assert output_physio.computed_metrics["respiratory_variance"].shape == (
        len(output_physio.data),
        2,
    )
    assert output_physio.computed_metrics["respiratory_variance_time"].shape == (
        len(output_physio.data),
        4,
    )
    assert output_physio.computed_metrics["cardiac_phase"].shape == (200, 20)

    # Exported metrics assertions
    assert os.path.exists(export_dir)

    assert os.path.exists(
        os.path.join(export_dir, "respiratory_variance_orig_convolved.1D")
    )
    assert os.path.exists(os.path.join(export_dir, "respiratory_variance_orig_raw.1D"))
    assert os.path.exists(
        os.path.join(export_dir, "respiratory_variance_resampled_convolved.1D")
    )
    assert os.path.exists(
        os.path.join(export_dir, "respiratory_variance_resampled_raw.1D")
    )
    assert os.path.exists(
        os.path.join(export_dir, "respiratory_variance_time_orig_convolved.1D")
    )
    assert os.path.exists(
        os.path.join(export_dir, "respiratory_variance_time_orig_raw.1D")
    )
    assert os.path.exists(
        os.path.join(export_dir, "respiratory_variance_time_resampled_convolved.1D")
    )
    assert os.path.exists(
        os.path.join(export_dir, "respiratory_variance_time_resampled_raw.1D")
    )
