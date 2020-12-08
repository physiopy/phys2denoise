import peakdet
import numpy as np
import matplotlib.pyplot as plt
from phys2denoise.metrics.rvt import rvt


def read_real_phys():
    real_physio = np.genfromtxt("../../data/sub-A00077637_ses-BAS1_task-rest_acq-1400_physio.tsv.gz")
    phys = peakdet.Physio(real_physio[:, 2]-real_physio[:, 2].mean(), fs=62.5)
    phys = peakdet.operations.filter_physio(phys, cutoffs=3, method="lowpass")
    return phys

def test_peakdet():
    phys = read_real_phys()
    phys = peakdet.operations.peakfind_physio(phys)
    assert phys.troughs is not None
    assert phys.peaks is not None

def test_rvt():
    phys = read_real_phys()
    phys = peakdet.operations.peakfind_physio(phys)
    r = rvt(phys.data, phys.peaks, phys.troughs, samplerate=phys.fs)
    plt.plot(r)
    plt.show()


