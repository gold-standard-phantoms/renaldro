import os
import argparse
import pdb
import shutil
from typing import List
import matplotlib.pyplot as plt
import imageio
from copy import deepcopy
from tempfile import TemporaryDirectory
from contextlib import nullcontext
import numpy as np
import nibabel as nib
from asldro.containers.image import NiftiImageContainer
from asldro.filters.resample_filter import ResampleFilter
from asldro.filters.nifti_loader import NiftiLoaderFilter
from asldro.pipelines.generate_ground_truth import generate_hrgt
from asldro.filters.bids_output_filter import BidsOutputFilter
from asldro.filters.load_asl_bids_filter import LoadAslBidsFilter
from asldro.pipelines.asl_quantification import asl_quantification
from asldro.utils.general import splitext
from asldro.cli import DirType
from asldro.examples import run_full_pipeline
from asldro.validators.user_parameter_input import (
    get_example_input_params,
    ECHO_TIME,
    REPETITION_TIME,
    ROT_Z,
    ROT_Y,
    ROT_X,
    TRANSL_Z,
    TRANSL_Y,
    TRANSL_X,
)
from renaldro.data.filepaths import (
    ASL_M0_DATA,
    ASL_RBF_DATA,
    HRGT_PARAMS,
    LABEL_MAP_DATA,
    T1_DATA,
)

TRANSF_PARAMS = [
    ROT_Z,
    ROT_Y,
    ROT_X,
    TRANSL_Z,
    TRANSL_Y,
    TRANSL_X,
]

ARRAY_PARAMS = [ECHO_TIME, REPETITION_TIME] + TRANSF_PARAMS

ARRAY_PARAM_DEFAULT = {
    ECHO_TIME: {"m0scan": 0.01, "control": 0.01, "label": 0.01,},
    REPETITION_TIME: {"m0scan": 10.0, "control": 5.0, "label": 5.0,},
    **{
        param: {"m0scan": 0.0, "control": 0.0, "label": 0.0,} for param in TRANSF_PARAMS
    },
}


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

    Outputs both a simple (where all values are assigned based on segmentation)
    and more realistic (where the perfusion rate and m0 derive from ASL data)
    ground truth data sets.

    Also then runs instances of ASLDRO's main pipeline for single subtraction
    and multiphase ASL


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

    # change the affine of the t1 image so that it is not oblique (this causes
    # problems for the TransformResampleImageFilter)
    # divide the affine by x,y, and z scaling to obtain just the rotation
    new_affine = np.eye(4)
    scaling_matrix = np.eye(4)
    scaling_matrix[:3, 0] = 1 / t1_image.voxel_size_mm[0]
    scaling_matrix[:3, 1] = 1 / t1_image.voxel_size_mm[1]
    scaling_matrix[:3, 2] = 1 / t1_image.voxel_size_mm[2]
    rot_affine = t1_image.affine * scaling_matrix
    # then rotate the translation part of the affine so it is aligned with the
    # xyz world axes
    new_affine[:, 3] = new_affine @ t1_image.affine[:, 3]
    new_affine[0, 0] = t1_image.voxel_size_mm[0]
    new_affine[1, 1] = t1_image.voxel_size_mm[1]
    new_affine[2, 2] = t1_image.voxel_size_mm[2]

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

    # set the affines of the seg_image, t1_image, rbf_image and m0_image to the
    # new affine
    t1_image.nifti_image.set_sform(new_affine)
    seg_image.nifti_image.set_sform(new_affine)
    rbf_image.nifti_image.set_sform(new_affine)
    m0_image.nifti_image.set_sform(new_affine)
    with TemporaryDirectory() as save_dir:
        seg_image_filename = os.path.join(save_dir, "seg_image.nii.gz")
        nib.save(seg_image.nifti_image, seg_image_filename)
        hrgt_simple = generate_hrgt(HRGT_PARAMS, seg_image_filename)
        hrgt_simple["name"] = "hrgt_simple"

    hrgt_real_rbf = deepcopy(hrgt_simple)
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
    hrgt_real_rbf["name"] = "hrgt_real_rbf"

    out = {
        "seg_image": seg_image,
        "t1_image": t1_image,
        "rbf_image": rbf_image,
        "m0_image": m0_image,
        "hrgt_simple": hrgt_simple,  # simple hrgt just with values applied
        "hrgt_real_rbf": hrgt_real_rbf,  # with rbf and m0 from the ASL
    }

    # context manager - if output_dir is None use TemporaryDirectory, otherwise
    # save directly to output_dir
    if output_dir is None:
        cm = TemporaryDirectory()
    else:
        cm = nullcontext(output_dir)

    with cm as save_dir:
        for hrgt in [hrgt_simple, hrgt_real_rbf]:
            for key in ["nifti", "json"]:
                hrgt[key] = os.path.join(
                    save_dir, hrgt["name"] + (".nii.gz" if key == "nifti" else ".json")
                )

            nib.save(hrgt["image"].nifti_image, hrgt["nifti"])
            BidsOutputFilter.save_json(hrgt["image_info"], hrgt["json"])

        for hrgt in [hrgt_simple, hrgt_real_rbf]:
            dro_params = generate_asldro_params(hrgt)
            dro_out_fn = os.path.join(save_dir, "dro_out_" + hrgt["name"] + ".zip")
            out["dro_out_" + hrgt["name"]] = run_full_pipeline(dro_params, dro_out_fn)
            out["dro_out_" + hrgt["name"]]["params"] = dro_params

            # unpack the archive
            dro_out_folder = os.path.join(save_dir, "dro_out_" + hrgt["name"])
            shutil.unpack_archive(dro_out_fn, dro_out_folder)

            # generate difference images for the single PLD data

            # asl data are in /sub-001/perf
            m0_fn = os.path.join(
                dro_out_folder, "sub-001", "perf", "sub-001_acq-001_m0scan.nii.gz"
            )
            m0_loader = NiftiLoaderFilter()
            m0_loader.add_input("filename", m0_fn)
            m0_loader.run()
            m0_data = np.squeeze(m0_loader.outputs["image"].image)
            m0_mask = m0_data > np.amax(m0_data) * 0.05  # threshold mask

            fn_base = {
                "single_pld": os.path.join(
                    dro_out_folder, "sub-001", "perf", "sub-001_acq-002_asl",
                ),
                "multi_pld": os.path.join(
                    dro_out_folder, "sub-001", "perf", "sub-001_acq-003_asl",
                ),
            }

            for series in ["single_pld", "multi_pld"]:
                asl_fn = os.path.join(fn_base[series] + ".nii.gz")
                asl_sidecar_fn = os.path.join(fn_base[series] + ".json")
                asl_context_fn = os.path.join(fn_base[series] + "context.tsv")
                asl_loader = LoadAslBidsFilter()
                asl_loader.add_input(LoadAslBidsFilter.KEY_IMAGE_FILENAME, asl_fn)
                asl_loader.add_input(
                    LoadAslBidsFilter.KEY_SIDECAR_FILENAME, asl_sidecar_fn
                )
                asl_loader.add_input(
                    LoadAslBidsFilter.KEY_ASLCONTEXT_FILENAME, asl_context_fn
                )
                asl_loader.run()

                # subtract to get the difference
                diff: NiftiImageContainer = asl_loader.outputs[
                    LoadAslBidsFilter.KEY_CONTROL
                ].clone()
                diff.image = (
                    asl_loader.outputs[LoadAslBidsFilter.KEY_CONTROL].image
                    - asl_loader.outputs[LoadAslBidsFilter.KEY_LABEL].image
                )
                # calculate Delta M / M0
                diff_norm = diff.clone()
                diff_norm.image = np.divide(
                    diff.image,
                    np.expand_dims(m0_data, axis=3)
                    if len(diff.shape) == 4
                    else m0_data,
                    out=np.zeros_like(diff.image),
                    where=np.expand_dims(m0_mask, axis=3)
                    if len(diff.shape) == 4
                    else m0_mask,
                )

                diff_fn = os.path.join(fn_base[series] + "_diff.nii.gz")
                diff_norm_fn = os.path.join(fn_base[series] + "_diff_normalised.nii.gz")
                nib.save(diff.nifti_image, diff_fn)
                nib.save(diff_norm.nifti_image, diff_norm_fn)

                # perform quantification of the single PLD dataset
                if series == "single_pld":
                    asl_quantification(
                        asl_fn,
                        output_dir=os.path.join(dro_out_folder, "sub-001", "perf"),
                    )
                    # load in the CBF file, apply the mask
                    cbf_fn = fn_base[series] + "_cbf.nii.gz"
                    cbf_loader = NiftiLoaderFilter()
                    cbf_loader.add_input("filename", cbf_fn)
                    cbf_loader.run()
                    cbf_image = cbf_loader.outputs["image"]
                    cbf_image.image = cbf_image.image * m0_mask
                    nib.save(cbf_image.nifti_image, cbf_fn)
                    # create a png image of the middle slice
                    f, ax = plt.subplots()
                    plt.imshow(
                        np.fliplr(np.rot90(cbf_image.image[:, :, 10])), cmap="plasma",
                    )
                    cbar = plt.colorbar()
                    cbar.set_label(r"RBF (ml/100g/min)")
                    plt.axis("off")
                    plt.axis([12, 52, 48, 16])
                    plt.show(block=False)
                    plt.savefig(fn_base[series] + "_cbf.png")
                    plt.close()

                # create a gif of the timeseries
                if series == "multi_pld":
                    nifti_timeseries_to_gif(
                        diff_norm,
                        10,
                        os.path.join(fn_base[series] + "_diff_normalised.gif"),
                        annotation_text=[
                            f"Label Duration = {diff.metadata['LabelingDuration']:.2f} \n"
                            f"PLD = {pld:.2f}"
                            for i, pld in enumerate(diff.metadata["PostLabelingDelay"])
                        ],
                        cbar_text=r"$\frac{\Delta M}{M0}$",
                    )

    return out


def generate_asldro_params(hrgt: dict) -> dict:
    """Creates parameters for ASL based on the supplied hrgt"""

    input_params = get_example_input_params()
    # add the hrgt's path to the configuration parameters
    input_params["global_configuration"]["ground_truth"] = {
        "nii": hrgt["nifti"],
        "json": hrgt["json"],
    }
    acq_matrix = [64, 64, 20]
    # copy the ground truth series to add at the end
    ground_truth_params = input_params["image_series"][
        [x["series_type"] for x in input_params["image_series"]].index("ground_truth")
    ]
    ground_truth_params["series_parameters"]["acq_matrix"] = acq_matrix
    # remove non-ASL series
    input_params["image_series"] = [
        x for x in input_params["image_series"] if x["series_type"] in ["asl"]
    ]

    ## M0 series
    input_params["image_series"][0]["series_description"] = "M0 " + hrgt["name"]
    input_params["image_series"][0]["series_parameters"][
        "background_suppression"
    ] = False
    input_params["image_series"][0]["series_parameters"]["desired_snr"] = 500
    input_params["image_series"][0]["series_parameters"]["acq_matrix"] = acq_matrix
    input_params["image_series"][0]["series_parameters"]["asl_context"] = "m0scan"

    ## Single PLD series comprising m0scan, control and label using the whitepaper model
    input_params["image_series"].append(deepcopy(input_params["image_series"][0]))
    input_params["image_series"][1]["series_description"] = (
        "ASL Single PLD " + hrgt["name"]
    )
    input_params["image_series"][1]["series_parameters"][
        "asl_context"
    ] = "m0scan control label"
    input_params["image_series"][1]["series_parameters"]["gkm_model"] = "whitepaper"

    ## Multi PLD series comprising control and label for PLD = 0 to 3.0s in 0.25s steps
    ## using the full general kinetic model
    input_params["image_series"].append(deepcopy(input_params["image_series"][0]))
    input_params["image_series"][2]["series_description"] = (
        "ASL Multi PLD " + hrgt["name"]
    )
    input_params["image_series"][2]["series_parameters"][
        "asl_context"
    ] = "control label"
    label_dur = input_params["image_series"][2]["series_parameters"]["label_duration"]
    input_params["image_series"][2]["series_parameters"]["signal_time"] = list(
        np.array(label_dur) + np.arange(0, 3.25, 0.25)
    )

    # set all the array parameters accordingly
    for n, _ in enumerate(input_params["image_series"]):
        asl_context = input_params["image_series"][n]["series_parameters"][
            "asl_context"
        ].split()
        # set the array parameters to their defaults (i.e. no motion etc)
        input_params["image_series"][n]["series_parameters"] = {
            **input_params["image_series"][n]["series_parameters"],
            **{
                param: [ARRAY_PARAM_DEFAULT[param][context] for context in asl_context]
                for param in ARRAY_PARAMS
            },
        }
    input_params["image_series"].append(ground_truth_params)

    return input_params


def nifti_timeseries_to_gif(
    image: NiftiImageContainer,
    slc: int,
    filename: str,
    colourmap: str = "plasma",
    annotation_text: List[str] = None,
    cbar_text: str = None,
    axis_lims: List[int] = None,
):
    with TemporaryDirectory() as gif_temp_dir:
        filenames = []
        num_images = image.shape[3]
        ny = image.shape[0]
        nx = image.shape[1]
        if annotation_text is not None:
            if len(annotation_text) != num_images:
                raise ValueError(
                    "number of annotations must match the timeseries length"
                )
        vmin = np.amin(image.image)
        vmax = np.amax(image.image)
        for idx in range(num_images):
            f, ax = plt.subplots()
            plt.imshow(
                np.fliplr(np.rot90(image.image[:, :, slc, idx])),
                vmin=vmin,
                vmax=vmax,
                cmap=colourmap,
            )
            cbar = plt.colorbar()
            cbar.set_label(cbar_text)
            plt.text(
                0,
                1.05,
                annotation_text[idx] if annotation_text is not None else None,
                transform=ax.transAxes,
            )
            plt.axis("off")
            plt.axis(axis_lims) if axis_lims is not None else None,
            plt.show(block=False)
            filenames.append(os.path.join(gif_temp_dir, f"{idx}.png"))
            plt.savefig(filenames[idx])
            plt.close()

        with imageio.get_writer(filename, mode="I", fps=1) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        # remove png images
        for filename in set(filenames):
            os.remove(filename)


if __name__ == "__main__":
    main()
