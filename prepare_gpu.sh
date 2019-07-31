#!/bin/bash
set -e
. ./prereq_install.sh

conda env create -f ./requirements_gpu.yml
conda activate mortgage_gpu

echo Measuring mortgage gpu
time python Mortgage_GPU.py  $DATASET_FOLDER 1
