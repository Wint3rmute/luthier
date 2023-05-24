import numpy as np
import pytest

from luthier.dsp import Sample


def test_mfcc_distance():
    s = Sample(np.zeros(100))
    assert s.mfcc_distance(s) == 0


def test_mfcc_distance_fails_on_different_sample_sizes():
    s1 = Sample(np.zeros(10))
    s2 = Sample(np.zeros(11))

    with pytest.raises(ValueError):
        s1.mfcc_distance(s2)


def test_spectrogram_distance():
    s = Sample(np.zeros(4096 * 10))
    assert s.spectrogram_distance(s) == 0
