"""filepaths.py tests"""

import os
from renaldro.data.filepaths import (
    ASL_M0_DATA,
    ASL_RBF_DATA,
    LABEL_MAP_DATA,
    T1_DATA,
)


def test_file_paths_exist():
    """Check that the files in data exist and are files"""
    for data in [ASL_M0_DATA, ASL_RBF_DATA, LABEL_MAP_DATA, T1_DATA]:
        assert all([os.path.isfile(data[key]) for key in data.keys()])
