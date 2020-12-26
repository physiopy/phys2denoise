import peakdet
import numpy as np
from phys2denoise.metrics.rvt import rvt


def read_fake_phys():
    f = 0.3
    fs = 62.5  # sampling rate
    t = 300
    samples = np.arange(t * fs) / fs
    noise = np.random.normal(0, 0.5, len(samples))
    fake_phys = 10 * np.sin(2 * np.pi * f * samples) + noise
    phys = peakdet.Physio(fake_phys, fs=62.5)
    phys = peakdet.operations.filter_physio(phys, cutoffs=3, method="lowpass")
    return phys


def test_peakdet():
    phys = read_fake_phys()
    phys = peakdet.operations.peakfind_physio(phys)
    assert phys.troughs is not None
    assert phys.peaks is not None


def test_rvt():
    phys = read_fake_phys()
    phys = peakdet.operations.peakfind_physio(phys)
    r = rvt(phys.data, phys.peaks, phys.troughs, samplerate=phys.fs)
    assert r is not None
    assert len(r) == 18750
