import numpy as np
import pytest


@pytest.fixture(scope="module")
def fake_phys():
    f = 0.3
    fs = 62.5  # sampling rate
    t = 300
    samples = np.arange(t * fs) / fs
    noise = np.random.normal(0, 0.5, len(samples))
    fake_phys = 10 * np.sin(2 * np.pi * f * samples) + noise
    return fake_phys
