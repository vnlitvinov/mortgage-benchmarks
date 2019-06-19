#!/bin/bash

. ./prereq_install.sh

conda env create -f ./requirements_gpu.yml
conda activate mortgage_gpu
