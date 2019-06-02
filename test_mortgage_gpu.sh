#!/bin/bash

mkdir test_mortgage_gpu && cd test_mortgage_gpu

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
rm Miniconda3-latest-Linux-x86_64.sh
export PATH="$HOME/miniconda/bin:$PATH"

conda create -n cudf_dev  -c nvidia -c rapidsai -c numba -c conda-forge -c defaults cudf=0.7 python=3.6 cudatoolkit=9.2 -y
conda init bash

mkdir dataset && cd dataset
wget http://rapidsai-data.s3-website.us-east-2.amazonaws.com/notebook-mortgage-data/mortgage_2000-2001.tgz
tar -xzvf mortgage_2000-2001.tgz
cd ../..

# now in mortgage-benchmarks folder
# time python Mortgage_GPU.py  test_mortgage_gpu/dataset/ 1

