#!/bin/bash

mkdir mortgage_gpu && cd mortgage_gpu

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"

conda create -n cudf_dev  -c nvidia -c rapidsai -c numba -c conda-forge -c defaults cudf=0.7 python=3.6 cudatoolkit=9.2 -y
conda init bash
source ~/.bashrc
conda activate cudf_dev
pip install xgboost

mkdir dataset && cd dataset
wget http://rapidsai-data.s3-website.us-east-2.amazonaws.com/notebook-mortgage-data/mortgage_2000-2001.tgz
tar -xzvf mortgage_2000-2001.tgz
cd ..

git clone https://github.com/vnlitvin/mortgage-benchmarks.git
cd mortgage-benchmarks
git checkout mortgage_gpu

time python Mortgage_GPU.py ../dataset/ 1

