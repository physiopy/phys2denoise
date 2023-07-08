import peakdet

from phys2denoise.metrics.chest_belt import respiratory_variance_time


def test_peakdet(fake_phys):
    phys = peakdet.Physio(fake_phys, fs=62.5)
    phys = peakdet.operations.filter_physio(phys, cutoffs=3, method="lowpass")
    phys = peakdet.operations.peakfind_physio(phys)
    assert phys.troughs is not None
    assert phys.peaks is not None


def test_respiratory_variance_time(fake_phys):
    phys = peakdet.Physio(fake_phys, fs=62.5)
    phys = peakdet.operations.filter_physio(phys, cutoffs=3, method="lowpass")
    phys = peakdet.operations.peakfind_physio(phys)
    r = test_respiratory_variance_time(
        phys.data, phys.peaks, phys.troughs, samplerate=phys.fs
    )
    assert r is not None
    assert len(r) == 18750
