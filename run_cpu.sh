#!/bin/bash


# prepare conda tool, mortgage dataset
. ./prereq_install.sh

conda env create -f requirements_cpu_daal.yml
conda activate mortgage_cpu_daal

# build pandas from source
git clone https://github.com/pandas-dev/pandas.git && cd pandas
python setup.py install
cd ..
