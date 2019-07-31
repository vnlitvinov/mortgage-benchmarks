#!/bin/bash


# prepare conda tool, mortgage dataset
. ./prereq_install.sh

conda env create -f requirements_cpu_daal.yml
echo "env mortgage_cpu_daal is created"

conda activate mortgage_cpu_daal
echo "mortgage_cpu_daal is activated"


# for build pandas from source uncomment the block below and
# add next lines to 'requirements_cpu_daal.yml':
# - cython
# - numpy=1.16.2

# git clone https://github.com/pandas-dev/pandas.git && cd pandas
#
# echo "building pandas from master"
# python setup.py install
# echo "build successful"

cd ..
echo "now you can run 'time python mortgage_pandas.py test_mortgage/mortgage_dataset/ 1 {daal, xgb}'"
