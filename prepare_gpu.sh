#!/bin/bash

. ./prereq_install.sh

conda env create -f ./requirements_gpu.yml
conda activate mortgage_gpu

echo "now you can run 'time python Mortgage_GPU.py  test_mortgage/mortgage_dataset/ 1'"
