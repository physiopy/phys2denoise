import numpy as np
import peakdet
import pytest
from _pytest.logging import LogCaptureFixture
from loguru import logger
from physutils import Physio

import phys2denoise.tasks as tasks


@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level=20,
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=False,
    )
    yield caplog
    logger.remove(handler_id)


@pytest.fixture(scope="module")
def fake_phys():
    f = 0.3
    fs = 62.5  # sampling rate
    t = 300
    samples = np.arange(t * fs) / fs
    noise = np.random.normal(0, 0.5, len(samples))
    fake_phys = 10 * np.sin(2 * np.pi * f * samples) + noise
    return fake_phys


@pytest.fixture(scope="module")
def fake_physio(fake_phys):
    phys = peakdet.Physio(fake_phys, fs=62.5)
    phys = peakdet.operations.filter_physio(phys, cutoffs=3, method="lowpass")
    phys = peakdet.operations.peakfind_physio(phys)

    # TODO: Change to a simpler call once physutils are
    # integrated to peakdet/prep4phys
    physio_obj = Physio(phys.data, phys.fs)
    physio_obj._metadata["peaks"] = phys.peaks
    physio_obj._metadata["troughs"] = phys.troughs

    return physio_obj


@pytest.fixture(scope="module")
def fake_physio_with_metrics(fake_physio):
    task = tasks.compute_metrics(
        phys=fake_physio,
        metrics=["respiratory_variance", "respiratory_variance_time", "cardiac_phase"],
        args={
            "cardiac_phase": {
                "slice_timings": np.linspace(0, 1, 22)[1:-1],
                "n_scans": 200,
                "t_r": 1.0,
            },
        },
    )

    task()
    new_physio = task.result().output.out

    return new_physio
