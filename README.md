# mortgage-benchmarks
A repo for benchmarking different implementations of ML stuff over Mortage data

Based on [benchmarks published by NVidia for their RAPIDS](https://render.githubusercontent.com/view/ipynb?commit=fd01c7ea360efe71745de540f3608843e147198e&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f72617069647361692f6e6f7465626f6f6b732f666430316337656133363065666537313734356465353430663336303838343365313437313938652f6d6f7274676167652f4532452e6970796e62&nwo=rapidsai%2Fnotebooks&path=mortgage%2FE2E.ipynb&repository_id=159398705&repository_type=Repository#Mortgage-Workflow).

# Run benchmarks

## GPU case
- launch `p3.2xlarge` gpu instance on AWS cloud with `Ubuntu Server 18.04 LTS (HVM), SSD Volume Type` image (this instance has a NVIDIA V100 GPU)
- login in it
- follow [the steps](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html#Cluster_GPUs_Manual_Install_Driver) to install the necessary driver for V100 GPU (using 396.82 version)
- `git clone https://github.com/vnlitvin/mortgage-benchmarks.git && cd mortgage-benchmarks`
- `. ./prepare_gpu.sh`
- `time python Mortgage_GPU.py  test_mortgage/mortgage_dataset/ 1`

## CPU case with using pandas upstream
- launch `c5.18xlarge` cpu instance on AWS cloud with `Ubuntu Server 18.04 LTS (HVM), SSD Volume Type` image
- login in it
- `git clone https://github.com/vnlitvin/mortgage-benchmarks.git && cd mortgage-benchmarks`
- `. ./prepare_cpu.sh`
- `time python mortgage_pandas.py test_mortgage/mortgage_dataset/ 1 daal` or
  `time python mortgage_pandas.py test_mortgage/mortgage_dataset/ 1 xgb`
