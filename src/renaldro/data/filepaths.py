"""Conastants with data file paths"""
import os

# the data directory in the renaldro module
DATA_DIR = os.path.dirname(os.path.realpath(__file__))

T1_DATA = {
    "json": os.path.join(DATA_DIR, "T1w.json"),
    "nifti": os.path.join(DATA_DIR, "T1w.nii.gz"),
}

ASL_M0_DATA = {
    "json": os.path.join(DATA_DIR, "ASL_M0.json"),
    "nifti": os.path.join(DATA_DIR, "ASL_M0.nii.gz"),
}

ASL_RBF_DATA = {
    "json": os.path.join(DATA_DIR, "ASL_RBF.json"),
    "nifti": os.path.join(DATA_DIR, "ASL_RBF.nii.gz"),
}

LABEL_MAP_DATA = {"nifti": os.path.join(DATA_DIR, "Label_Map_cortex_medulla.nii.gz")}
