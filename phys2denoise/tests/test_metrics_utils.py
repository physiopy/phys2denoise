import numpy as np
import pytest

from phys2denoise.metrics.utils import mirrorpad_1d, rms_envelope_1d


@pytest.fixture
def short_arr():
    return np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


def test_mirrorpad(short_arr):
    """Basic test for flipping and padding the input array"""
    arr_mirror = mirrorpad_1d(short_arr, buffer=3)
    expected_arr_mirror = np.array([2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 7])
    assert all(arr_mirror == expected_arr_mirror)


def test_mirrorpad_exception(short_arr, caplog):
    """When passing array that is too short to perform mirrorpadding, the
    function should give an error."""
    arr = np.array(short_arr)
    arr_mirror = mirrorpad_1d(arr)
    assert caplog.text.count("Fixing buffer size to array length.") > 0


def test_rms_envelope():
    """Basic test for rms envelope calculation. When the input is constant, we
    should get an output with the same size and the same constant value."""
    arr = np.tile(2.0, 10)
    arr_env = rms_envelope_1d(arr, window=4)
    expected_arr_env = np.tile(2.0, 10)
    assert all(arr_env == expected_arr_env)
