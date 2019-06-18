#!/bin/bash

. ./prereq_install.sh
conda env create -f requirements_cpu_daal.yml
conda activate mortgage_cpu_daal

