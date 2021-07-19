import os
import pytest
import nibabel as nib
import numpy.testing
from renaldro.main import main
from renaldro.data.filepaths import DATA_DIR


def test_main():
    out = main()
    t1_image = out["t1_image"]
    seg_image = out["seg_image"]
    rbf_image = out["rbf_image"]
    m0_image = out["m0_image"]
    # check that the seg_image's affine has been correctly set
    numpy.testing.assert_array_equal(seg_image.affine, t1_image.affine)
    assert seg_image.space_units == t1_image.space_units
    assert seg_image.time_units == t1_image.time_units
    numpy.testing.assert_array_equal(seg_image.voxel_size_mm, t1_image.voxel_size_mm)

    # check that the resampled rbf is correct
    numpy.testing.assert_array_equal(rbf_image.affine, t1_image.affine)

    #check that the resampled m0 is correct
    numpy.testing.assert_array_equal(m0_image.affine, t1_image.affine)

    # ib.save(seg_image.nifti_image, os.path.join(DATA_DIR, "corrected_seg.nii.gz"))
    # nib.save(rbf_image.nifti_image, os.path.join(DATA_DIR, "resampled_rbf.nii.gz"))
