import os
import argparse
from copy import deepcopy
from tempfile import TemporaryDirectory
import numpy as np
import nibabel as nib
from asldro.containers.image import NiftiImageContainer
from asldro.filters.resample_filter import ResampleFilter
from asldro.filters.nifti_loader import NiftiLoaderFilter
from asldro.pipelines.generate_ground_truth import generate_hrgt
from asldro.filters.bids_output_filter import BidsOutputFilter
from asldro.cli import DirType
from renaldro.data.filepaths import (
    ASL_M0_DATA,
    ASL_RBF_DATA,
    HRGT_PARAMS,
    LABEL_MAP_DATA,
    T1_DATA,
)


def main():
    """handles parsing of the command line interface then calls generate_renal_ground_truth"""
    parser = argparse.ArgumentParser(
        description="""Generation of a renal ground truth
        for ASLDRO"""
    )

    parser.set_defaults(func=lambda _: parser.print_help())
    parser.add_argument(
        "output_dir",
        type=DirType(should_exist=True),
        help="A path to a directory to save the output files to, must exist",
    )
    args = parser.parse_args()

    generate_renal_ground_truth(args.output_dir)


def generate_renal_ground_truth(output_dir: str = None) -> dict:
    """Generates the Renal ground truth for ASLDRO

    Outputs both a simple (where all values are assig ned based on segmentation)
    and more realistic (where the perfusion rate and m0 derive from ASL data)
    ground truth data sets.

    :param output_dir: Directory to output data to, defaults to None
    :type output_dir: str, optional
    :return: the output and intermediate images in a dictionary
    :rtype: dict
    """

    # Load in the segmentation image
    seg_loader = NiftiLoaderFilter()
    seg_loader.add_input("filename", LABEL_MAP_DATA["nifti"])
    seg_loader.run()
    seg_image: NiftiImageContainer = seg_loader.outputs["image"]
    # load in the t1 image
    t1_loader = NiftiLoaderFilter()
    t1_loader.add_input("filename", T1_DATA["nifti"])
    t1_loader.run()
    t1_image: NiftiImageContainer = t1_loader.outputs["image"]

    # set the affine of the seg_image to be the same as the affine of  t1_image
    seg_image.nifti_image.set_sform(t1_image.affine)
    # set the space and time units
    seg_image.space_units = t1_image.space_units
    seg_image.time_units = t1_image.time_units
    seg_image.voxel_size_mm = t1_image.voxel_size_mm

    # the seg image also needs flipping in the third axis
    seg_image.image = np.flip(seg_image.image, axis=1)

    # load in the rbf data
    rbf_loader = NiftiLoaderFilter()
    rbf_loader.add_input("filename", ASL_RBF_DATA["nifti"])

    m0_loader = NiftiLoaderFilter()
    m0_loader.add_input("filename", ASL_M0_DATA["nifti"])

    # resample to match the t1_image
    rbf_resampler = ResampleFilter()
    rbf_resampler.add_parent_filter(rbf_loader)
    rbf_resampler.add_input(ResampleFilter.KEY_AFFINE, t1_image.affine)
    rbf_resampler.add_input(ResampleFilter.KEY_SHAPE, t1_image.shape)
    rbf_resampler.add_input(ResampleFilter.KEY_INTERPOLATION, ResampleFilter.LINEAR)
    rbf_resampler.run()
    rbf_image: NiftiImageContainer = rbf_resampler.outputs["image"]

    m0_resampler = ResampleFilter()
    m0_resampler.add_parent_filter(m0_loader)
    m0_resampler.add_input(ResampleFilter.KEY_AFFINE, t1_image.affine)
    m0_resampler.add_input(ResampleFilter.KEY_SHAPE, t1_image.shape)
    m0_resampler.add_input(ResampleFilter.KEY_INTERPOLATION, ResampleFilter.LINEAR)
    m0_resampler.run()
    m0_image: NiftiImageContainer = m0_resampler.outputs["image"]

    with TemporaryDirectory() as temp_dir:
        seg_image_filename = os.path.join(temp_dir, "seg_image.nii.gz")
        nib.save(seg_image.nifti_image, seg_image_filename)
        hrgt = generate_hrgt(HRGT_PARAMS, seg_image_filename)

    hrgt_real_rbf = deepcopy(hrgt)
    # place the rbf map into the hrgt's "perfusion_rate" image
    hrgt_real_rbf["image"].image[
        :,
        :,
        :,
        :,
        list(hrgt_real_rbf["image_info"]["quantities"]).index("perfusion_rate"),
    ] = np.expand_dims(np.atleast_3d(rbf_image.image), axis=3)

    # place the m0 map into the hrgt's "m0" image
    hrgt_real_rbf["image"].image[
        :, :, :, :, list(hrgt_real_rbf["image_info"]["quantities"]).index("m0")
    ] = np.expand_dims(np.atleast_3d(m0_image.image), axis=3)

    out = {
        "seg_image": seg_image,
        "t1_image": t1_image,
        "rbf_image": rbf_image,
        "m0_image": m0_image,
        "hrgt_simple": hrgt,  # simple hrgt just with values applied
        "hrgt_real_rbf": hrgt_real_rbf,  # with rbf and m0 from the ASL
    }

    if output_dir is not None:
        out["hrgt_simple"]["nifti"] = os.path.join(
            output_dir, "renal_hrgt_simple.nii.gz"
        )
        out["hrgt_simple"]["json"] = os.path.join(output_dir, "renal_hrgt_simple.json")
        out["hrgt_real_rbf"]["nifti"] = os.path.join(
            output_dir, "renal_hrgt_real_rbf.nii.gz"
        )
        out["hrgt_real_rbf"]["json"] = os.path.join(
            output_dir, "renal_hrgt_real_rbf.json"
        )

        nib.save(
            hrgt["image"].nifti_image, out["hrgt_simple"]["nifti"],
        )
        nib.save(
            hrgt_real_rbf["image"].nifti_image, out["hrgt_real_rbf"]["nifti"],
        )
        BidsOutputFilter.save_json(hrgt["image_info"], out["hrgt_simple"]["json"])
        BidsOutputFilter.save_json(
            hrgt_real_rbf["image_info"], out["hrgt_real_rbf"]["json"],
        )

    return out


if __name__ == "__main__":
    main()
