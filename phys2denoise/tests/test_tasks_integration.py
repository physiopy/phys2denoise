"""Tests for phys2denoise.tasks and their integration."""
import os

import nest_asyncio
import numpy as np
from loguru import logger
from physutils import physio
from physutils.tasks import transform_to_physio
from pydra import Submitter, Workflow

import phys2denoise.tasks as tasks

nest_asyncio.apply()


def test_integration(fake_phys):
    """Test the integration of phys2denoise tasks."""
    file_dir = os.path.dirname(os.path.abspath(__file__))
    export_dir = os.path.join(file_dir, "test_output_data/integration")

    wf = Workflow(
        name="metrics_wf",
        input_spec=["phys", "fs"],
        phys="phys2denoise/tests/data/ECG.csv",
        fs=62.5,
    )
    wf.add(
        transform_to_physio(
            name="transform_to_physio", phys=wf.lzin.phys, fs=wf.lzin.fs, mode="physio"
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
    wf.set_output([("result", wf.export_metrics.lzout.out)])

    with Submitter(plugin="cf") as sub:
        sub(wf)
    wf().result()

    assert os.path.exists(export_dir)
