import numpy as np
import nibabel as nib
from asldro.containers.image import NiftiImageContainer
from asldro.filters.resample_filter import ResampleFilter
from asldro.filters.nifti_loader import NiftiLoaderFilter

from renaldro.data.filepaths import ASL_M0_DATA, ASL_RBF_DATA, LABEL_MAP_DATA, T1_DATA


def main() -> dict:

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
    rbf_image = rbf_resampler.outputs["image"]

    m0_resampler = ResampleFilter()
    m0_resampler.add_parent_filter(m0_loader)
    m0_resampler.add_input(ResampleFilter.KEY_AFFINE, t1_image.affine)
    m0_resampler.add_input(ResampleFilter.KEY_SHAPE, t1_image.shape)
    m0_resampler.add_input(ResampleFilter.KEY_INTERPOLATION, ResampleFilter.LINEAR)
    m0_resampler.run()
    m0_image = m0_resampler.outputs["image"]

    return {
        "seg_image": seg_image,
        "t1_image": t1_image,
        "rbf_image": rbf_image,
        "m0_image": m0_image,
    }


if __name__ == "__main__":
    main()
