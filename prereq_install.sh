#!/bin/bash


# functions
function create_dir {
    FOLDER=$1
    if [[ ! -d $FOLDER ]]; then
        echo "$FOLDER folder wasn't found"
        mkdir $FOLDER
        echo "$FOLDER folder is created"
    else
        echo "$FOLDER folder is existing"
    fi
    echo ""
}


# constants
TEST_FOLDER="$(pwd)/test_mortgage"
DATASET_FOLDER="$TEST_FOLDER/mortgage_dataset"
DATASET_LINK="http://rapidsai-data.s3-website.us-east-2.amazonaws.com/notebook-mortgage-data/mortgage_2000-2001.tgz"
DATASET_NAME="$DATASET_FOLDER/mortgage_2000-2001.tgz"


sudo apt-get update -y
sudo apt-get install gcc g++ -y


create_dir $TEST_FOLDER
cd $TEST_FOLDER


if [[ ! "$(command -v conda)" ]]; then
    echo "conda command wasn't found in the PATH; attemp to install miniconda"
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
    rm Miniconda3-latest-Linux-x86_64.sh
    export PATH="$HOME/miniconda/bin:$PATH"
    conda init bash
    source ~/.bashrc
    echo "conda installed"
else
    echo "conda command was found in the PATH"
fi
echo ""


create_dir $DATASET_FOLDER
cd $DATASET_FOLDER


if [[ ! -f $DATASET_NAME ]]; then
    echo "$DATASET_NAME file wasn't found"
    echo "attemp to download this file from $DATASET_LINK"
    wget $DATASET_LINK
    tar -xzvf $DATASET_NAME
else
    echo "$DATASET_NAME file was found"
fi
cd ../..

