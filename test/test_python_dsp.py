import numpy as np
import pytest

from luthier.dsp import Sample


def test_mfcc_distance() -> None:
    s = Sample(np.zeros(2048))
    assert s.mfcc_distance(s) == 0


def test_mfcc_distance_fails_on_different_sample_sizes() -> None:
    s1 = Sample(np.zeros(4096))
    s2 = Sample(np.zeros(4097))

    with pytest.raises(ValueError):
        s1.mfcc_distance(s2)


def test_spectrogram_distance() -> None:
    s = Sample(np.zeros(4096 * 10))
    assert s.spectrogram_distance(s) == 0


def test_mfcc_dist_with_rms() -> None:
    s = Sample(np.zeros(4096 * 10))
    assert s.mfcc_distance_with_rms(s) == 0


def test_spectrogram_dist_with_dtw() -> None:
    s = Sample(np.zeros(4096 * 10))
    assert s.spectrogram_distance_with_dtw(s) == 0


def test_mfcc_dist_with_rms_and_dtw() -> None:
    s = Sample(np.zeros(4096 * 10))
    assert s.mfcc_distance_with_dtw_and_rms(s) == 0


def test_various_dtw_window_lengths() -> None:
    s = Sample(np.zeros(4096 * 10))

    for w in range(60):
        s.mfcc_distance_with_dtw(s, w=w * 4 - 10)
