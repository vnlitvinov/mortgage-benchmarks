#!/bin/bash

sudo apt-get update
sudo apt-get install gcc g++

. ./prereq_install.sh

conda env create -f requirements_cpu_daal.yml
conda activate mortgage_cpu_daal

