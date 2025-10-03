This install took a special mix for the cuda compilers for the helpix grid. 

1) conda create -n cbottle
2) conda activate cbottle
3) conda install -c conda-forge python=3.12 pip 
4) /anaconda/envs/cbottle/bin/pip install torch 
5) conda install -c conda-forge "cuda-toolkit=12.2" -y
6) 
export CUDA_HOME="$CONDA_PREFIX"
export CUDAHOME="$CONDA_PREFIX"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export CUDAHOSTCXX="$CXX"

7) /anaconda/envs/cbottle/bin/pip install -U pip setuptools wheel scikit-build-core ninja
8) /anaconda/envs/cbottle/bin/pip cache purge
9) /anaconda/envs/cbottle/bin/pip install --no-build-isolation https://github.com/NVlabs/earth2grid/archive/main.tar.gz
10) /anaconda/envs/cbottle/bin/pip install earth2studio[dlwp]
11) /anaconda/envs/cbottle/bin/pip install ipykernel 
12) python -m ipykernel install --user --name cbottle --display-name "cbottle"

#for file transfer from azure 
13) /anaconda/envs/cbottle/bin/pip install azure-ai-ml 
14) /anaconda/envs/cbottle/bin/pip install azure-identity 

