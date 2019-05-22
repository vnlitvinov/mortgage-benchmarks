# mortgage-benchmarks
A repo for benchmarking different implementations of ML stuff over Mortage data

Based on [benchmarks published by NVidia for theri RAPIDS](https://render.githubusercontent.com/view/ipynb?commit=fd01c7ea360efe71745de540f3608843e147198e&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f72617069647361692f6e6f7465626f6f6b732f666430316337656133363065666537313734356465353430663336303838343365313437313938652f6d6f7274676167652f4532452e6970796e62&nwo=rapidsai%2Fnotebooks&path=mortgage%2FE2E.ipynb&repository_id=159398705&repository_type=Repository#Mortgage-Workflow).

# Installation from conda

```
# for CUDA 9.2
conda create -n cudf_dev  -c nvidia -c rapidsai -c numba -c conda-forge -c defaults \
    cudf=0.7 python=3.6 cudatoolkit=9.2
```

# Getting dataset

wget http://rapidsai-data.s3-website.us-east-2.amazonaws.com/notebook-mortgage-data/mortgage_2000-2001.tgz

tar -xzvf mortgage_2000-2001.tgz
