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

## Changelog

[1.0.0] = 02-08-2021

Added:

- Project repository set up with Continuous Integration
- Data from iBEAt study
- Generation of simple and realistic ground truths
- Generation of DRO data for single and multi PLD
- Calculation of RBF using single PLD data.
- Generation of figures for Parenchima 2021 meeting abstract
