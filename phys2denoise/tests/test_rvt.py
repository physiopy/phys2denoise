import peakdet
from physutils import io, physio

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

    # TODO: Change to a simpler call once physutils are
    # integrated to peakdet/prep4phys
    r = respiratory_variance_time(
        phys.data, fs=phys.fs, peaks=phys.peaks, troughs=phys.troughs
    )
    assert r is not None
    assert len(r) == 18750


def test_respiratory_variance_time(fake_phys):
    phys = peakdet.Physio(fake_phys, fs=62.5)
    phys = peakdet.operations.filter_physio(phys, cutoffs=3, method="lowpass")
    phys = peakdet.operations.peakfind_physio(phys)

    # TODO: Change to a simpler call once physutils are
    # integrated to peakdet/prep4phys
    r = respiratory_variance_time(
        phys.data, fs=phys.fs, peaks=phys.peaks, troughs=phys.troughs
    )
    assert r is not None
    assert len(r) == 18750


def test_respiratory_variance_time_physio_obj(fake_phys):
    phys = peakdet.Physio(fake_phys, fs=62.5)
    phys = peakdet.operations.filter_physio(phys, cutoffs=3, method="lowpass")
    phys = peakdet.operations.peakfind_physio(phys)

    # TODO: Change to a simpler call once physutils are
    # integrated to peakdet/prep4phys
    physio_obj = physio.Physio(phys.data, phys.fs)
    physio_obj._metadata["peaks"] = phys.peaks
    physio_obj._metadata["troughs"] = phys.troughs
    physio_obj = respiratory_variance_time(physio_obj)

    assert (
        physio_obj.history[0][0]
        == "phys2denoise.metrics.chest_belt.respiratory_variance_time"
    )
    assert physio_obj.computed_metrics["respiratory_variance_time"].ndim == 2
    assert physio_obj.computed_metrics["respiratory_variance_time"].shape == (18750, 4)
    assert physio_obj.computed_metrics["respiratory_variance_time"].has_lags == True

    # assert r is not None
    # assert len(r) == 18750
