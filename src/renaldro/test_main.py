import sys
import numpy as np
import json
import os
from asldro.filters.json_loader import JsonLoaderFilter
from asldro.filters.nifti_loader import NiftiLoaderFilter
from asldro.filters.transform_resample_image_filter import TransformResampleImageFilter
import pytest
from unittest.mock import patch
from tempfile import TemporaryDirectory
import nibabel as nib
import numpy.testing
from renaldro.main import main, generate_renal_ground_truth, generate_asldro_params
from renaldro.data.filepaths import DATA_DIR


def test_generate_renal_ground_truth():
    """Tests the generation of the renal DRO"""
    with TemporaryDirectory() as temp_dir:
        out = generate_renal_ground_truth(temp_dir)
        t1_image = out["t1_image"]
        seg_image = out["seg_image"]
        rbf_image = out["rbf_image"]
        m0_image = out["m0_image"]
        hrgt_simple = out["hrgt_simple"]
        hrgt_real_rbf = out["hrgt_real_rbf"]
        # check that the seg_image's affine has been correctly set
        numpy.testing.assert_array_equal(seg_image.affine, t1_image.affine)
        assert seg_image.space_units == t1_image.space_units
        assert seg_image.time_units == t1_image.time_units
        numpy.testing.assert_array_equal(
            seg_image.voxel_size_mm, t1_image.voxel_size_mm
        )

        # check that the resampled rbf is correct
        numpy.testing.assert_array_equal(rbf_image.affine, t1_image.affine)

        # check that the resampled m0 is correct
        numpy.testing.assert_array_equal(m0_image.affine, t1_image.affine)

        for hrgt in ["hrgt_simple", "hrgt_real_rbf"]:
            nifti_loader = NiftiLoaderFilter()
            nifti_loader.add_input("filename", out[hrgt]["nifti"])
            nifti_loader.run()
            json_loader = JsonLoaderFilter()
            json_loader.add_input("filename", out[hrgt]["json"])
            json_loader.run()

            assert out[hrgt]["image_info"] == {
                "quantities": [
                    "perfusion_rate",
                    "transit_time",
                    "t1",
                    "t2",
                    "t2_star",
                    "m0",
                    "seg_label",
                ],
                "units": ["ml/100g/min", "s", "s", "s", "s", "", ""],
                "segmentation": {
                    "background": 0,
                    "cortex": 1,
                    "medulla": 2,
                },
                "parameters": {
                    "lambda_blood_brain": 0.9,
                    "magnetic_field_strength": 3.0,
                    "t1_arterial_blood": 1.65,
                },
            }
            assert json_loader.outputs == out[hrgt]["image_info"]

            numpy.testing.assert_array_equal(
                nifti_loader.outputs["image"].image, out[hrgt]["image"].image
            )

            assert os.path.exists(os.path.join(temp_dir, "dro_out_ss_hrgt_simple.zip"))
            assert os.path.exists(
                os.path.join(temp_dir, "dro_out_ss_hrgt_real_rbf.zip")
            )

            asl_simple = out["dro_out_hrgt_simple"]["asldro_output"][0]

            gt_m0 = out["dro_out_hrgt_simple"]["hrgt"]["m0"]


def test_main_cli():
    """Tests the command line interface"""
    with TemporaryDirectory() as temp_dir:
        testargs = ["renaldro", str(temp_dir)]
        with patch.object(sys, "argv", testargs):
            main()


def test_generate_asldro_params():
    """Tests the single subtraction parameter generation function"""

    hrgt = {
        "nifti": "path/to/hrgt.nii.gz",
        "json": "path/to/hrgt.json",
        "name": "random hrgt",
    }
    params = generate_asldro_params(hrgt)

    assert params["global_configuration"]["ground_truth"] == {
        "nii": "path/to/hrgt.nii.gz",
        "json": "path/to/hrgt.json",
    }
    assert params["image_series"][0]["series_type"] == "asl"
