#!/bin/bash


# prepare conda tool, mortgage dataset
. ./prereq_install.sh

conda env create -f requirements_cpu_daal.yml
echo "env mortgage_cpu_daal is created"

echo "conda activate mortgage_cpu_daal"
conda activate mortgage_cpu_daal
echo "mortgage_cpu_daal is activated"

# build pandas from source
git clone https://github.com/pandas-dev/pandas.git && cd pandas

echo "building pandas from master"
python setup.py install
echo "build successful"

cd ..
