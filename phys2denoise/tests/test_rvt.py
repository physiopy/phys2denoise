import peakdet
import numpy as np
import pytest
from phys2denoise.metrics.chest_belt import rvt


@pytest.fixture
def fake_phys():
    f = 0.3
    fs = 62.5  # sampling rate
    t = 300
    samples = np.arange(t * fs) / fs
    noise = np.random.normal(0, 0.5, len(samples))
    fake_phys = 10 * np.sin(2 * np.pi * f * samples) + noise
    phys = peakdet.Physio(fake_phys, fs=62.5)
    phys = peakdet.operations.filter_physio(phys, cutoffs=3, method="lowpass")
    return phys


def test_peakdet(fake_phys):
    phys = peakdet.operations.peakfind_physio(fake_phys)
    assert phys.troughs is not None
    assert phys.peaks is not None


def test_rvt(fake_phys):
    phys = peakdet.operations.peakfind_physio(fake_phys)
    r = rvt(phys.data, phys.peaks, phys.troughs, samplerate=phys.fs)
    assert r is not None
    assert len(r) == 18750
