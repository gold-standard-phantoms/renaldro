# renaldro

Creation of a renal DRO for ASLDRO. Supports python 3.7 and newer.

## Set up

Set up the python `virtualenv` and install as a module

Pure python:

    virtualenv env -p python3 --clear
    source env/bin/activate
    pip install -e .

Anaconda:

    conda create --name renaldro python=3.7
    activate renaldro
    pip install -e .

## How to run

Take a single command line argument which is the directory to output files to:

    renaldro /path/to/output_directory
