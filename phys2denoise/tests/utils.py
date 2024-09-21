import logging
import os

import numpy as np
import peakdet
from physutils import Physio, io

LGR = logging.getLogger(__name__)
LGR.setLevel(logging.DEBUG)


def create_fake_phys():
    f = 0.3
    fs = 62.5  # sampling rate
    t = 300
    samples = np.arange(t * fs) / fs
    noise = np.random.normal(0, 0.5, len(samples))
    fake_phys = 10 * np.sin(2 * np.pi * f * samples) + noise

    phys = peakdet.Physio(fake_phys, fs=62.5)
    phys = peakdet.operations.filter_physio(phys, cutoffs=3, method="lowpass")
    phys = peakdet.operations.peakfind_physio(phys)

    # TODO: Change to a simpler call once physutils are
    # integrated to peakdet/prep4phys
    physio_obj = Physio(phys.data, phys.fs)
    physio_obj._metadata["peaks"] = phys.peaks
    physio_obj._metadata["troughs"] = phys.troughs

    LGR.debug(f"Current working directory: {os.getcwd()}")
    save_path = os.path.abspath("phys2denoise/tests/data/fake_phys.phys")
    LGR.info(f"save_path: {save_path}")

    io.save_physio(save_path, physio_obj)

    return physio_obj
