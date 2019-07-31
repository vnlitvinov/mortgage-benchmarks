#!/bin/bash
set -e

# prepare conda tool, mortgage dataset
source ./prereq_install.sh

conda env list | grep -q mortgage_cpu_daal || conda env create -f requirements_cpu_daal.yml
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


cd `dirname $0`
echo Measuring with DAAL4PY
time python mortgage_pandas.py $DATASET_FOLDER 1 daal
echo Measuring with XGBoost
time python mortgage_pandas.py $DATASET_FOLDER 1 xgb

