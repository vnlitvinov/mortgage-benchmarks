# mortgage-benchmarks
A repo for benchmarking different implementations of ML stuff over Mortage data

Based on [benchmarks published by NVidia for theri RAPIDS](https://render.githubusercontent.com/view/ipynb?commit=fd01c7ea360efe71745de540f3608843e147198e&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f72617069647361692f6e6f7465626f6f6b732f666430316337656133363065666537313734356465353430663336303838343365313437313938652f6d6f7274676167652f4532452e6970796e62&nwo=rapidsai%2Fnotebooks&path=mortgage%2FE2E.ipynb&repository_id=159398705&repository_type=Repository#Mortgage-Workflow).

# Run benchmarks
- launch `p3.2xlarge` gpu instance on AWS cloud (this instance has a NVIDIA V100 GPU)
- login in it
- follow [the steps](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html#Cluster_GPUs_Manual_Install_Driver) to install the necessary driver for V100 GPU (using 396.82 version)
- `git clone https://github.com/vnlitvin/mortgage-benchmarks.git`
- `cd mortgage-benchmarks`
- `git checkout mortgage_gpu`
- `chmod +x test_mortgage_gpu.sh && ./test_mortgage_gpu.sh`
- `bash`
- `conda activate cudf_dev`
- `pip install xgboost`
- `time python Mortgage_GPU.py  test_mortgage_gpu/dataset/ 1`

# Results for one quarter
- ETL time:  8.85 sec.
- Machine learning - train:  35.13 sec.
