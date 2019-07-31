#!/bin/bash
set -e
source ./prereq_install.sh

conda env list | grep -q mortgage_gpu || conda env create -f ./requirements_gpu.yml
conda activate mortgage_gpu

echo ''
echo -------------/ Measuring mortgage gpu /-------------
echo ''
time python Mortgage_GPU.py  $DATASET_FOLDER 1
