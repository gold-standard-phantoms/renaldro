import pytest
import numpy.testing
from renaldro.main import main


def test_main():
    out = main()
    # check that the seg_image's affine has been correctly set
    numpy.testing.assert_array_equal(out["seg_image"].affine, out["t1_image"].affine)
    assert out["seg_image"].space_units == out["t1_image"].space_units
    assert out["seg_image"].time_units == out["t1_image"].time_units
    numpy.testing.assert_array_equal(
        out["seg_image"].voxel_size_mm, out["t1_image"].voxel_size_mm
    )
