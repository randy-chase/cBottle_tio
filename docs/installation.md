# Installation

## PIP

If necessary install pytorch. Then, we need to install earth2grid
```
pip install torch # if necessary
pip install setuptools hatchling
pip install --no-build-isolation https://github.com/NVlabs/earth2grid/archive/main.tar.gz
```

**Warning: make sure to include `--no-build-isolation` above to avoid building earth2grid against the wrong version of pytorch. This will cause very confusing error messages during runtime.**

Now, install cbottle
```
pip install -e .
```

## NERSC (instructions)

Install the pre-requisites
```
module load pytorch/2.6.0
python3 -m venv --system-site-packages cbottle-env
source cbottle-env/bin/activate
CC=gcc CXX=g++ pip install --no-build-isolation https://github.com/NVlabs/earth2grid/archive/main.tar.gz
```

Then clone cBottle and install some other dependencies
```
git clone https://github.com/NVlabs/cBottle
cd cBottle
pip install -e .
```

Then, you will need to configure NERSC paths using a .env file (you can also manually export the environment variables)
```
SUBMIT_ACCOUNT=
SUBMIT_SCRIPT=../../ord_scripts/submit_ord.sh

V6_ERA5_ZARR=/global/cfs/cdirs/trn006/data/nvidia/era5_hpx_6.zarr/
RAW_DATA_URL_7=s3://ICON_cycle3_ngc3028/ngc3028_PT30M_7.zarr/
RAW_DATA_URL_6=/global/cfs/cdirs/trn006/data/nvidia/ngc3028_PT30M_6.zarr/
RAW_DATA_URL_4=s3://ICON_cycle3_ngc3028/ngc3028_PT30M_4.zarr/
RAW_DATA_URL=/global/cfs/cdirs/trn006/data/nvidia/ngc3028_PT30M_4weeks_10.zarr/

V6_ICON_ZARR=/global/cfs/cdirs/trn006/data/nvidia/ICON_v6_dataset.zarr/
V6_ICON_ZARR_PROFILE=
RAW_DATA_PROFILE=
SST_MONMEAN_DATA_PROFILE=
LAND_DATA_PROFILE=

LAND_DATA_URL_10=/global/cfs/cdirs/trn006/data/nvidia/landfraction/ngc3028_P1D_10.zarr/
LAND_DATA_URL_6=/global/cfs/cdirs/trn006/data/nvidia/landfraction/ngc3028_P1D_6.zarr/
LAND_DATA_URL_4=s3://ICON_cycle3_ngc3028/landfraction/ngc3028_P1D_4.zarr/

SST_MONMEAN_DATA_URL_6=/global/cfs/cdirs/trn006/data/nvidia/ngc3028_P1D_ts_monmean_6.zarr
SST_MONMEAN_DATA_URL_4=

ERA5_HPX64_PATH=
ERA5_NPY_PATH_4=

AMIP_MID_MONTH_SST=s3://input4MIPs/tosbcs_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-9_gn_187001-202212.nc
```

At this point you should be able to run the training and other inference commands.
You will want a GPU node, this can be requested like this:
```
srun --nodes 1 --qos interactive --time 04:00:00 -C 'gpu&hbm80g' --gpus 1 --account=trn006  --pty /bin/bash
```
