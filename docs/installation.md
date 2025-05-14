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

### NERSC interactive session

To get an interactive session run

```
scripts/nersc/env/interactive.sh
```

This will request a single 80 Gb A100 and also configure the environment to
point to point to the datasets at NERSC. 

> [!NOTE]
> If you are interested, you can see the configured options in the `scripts/nersc/env` file. These are loaded in [this python file](../src/cbottle/config/environment.py).

Once you have obtained the interactive session, you can run the coarse-resolution training like this

```
# Source your venv again
source ~/cbottle-env/bin/activate

# modify these as desired
MY_OUTPUT_PATH=path/to/the/output
NAME=my_experiment

# launch the training
python3 scripts/train_coarse.py --loop.noise_distribution log_uniform \
--loop.sigma_min 0.02 --loop.sigma_max 200 --loop.label_dropout 0.25 \
--loop.batch_gpu 4 --loop.batch_size 64 --loop.dataloader_num_workers 8 \
--loop.with_era5 --loop.use_labels  --loop.data_version 6 \
--loop.monthly_sst_input --name v6data  --loop.dataloader_prefetch_factor 100 \
--output_dir $MY_OUTPUT_PATH --name $NAME
```

The outputed checkpoints will be stored at "$MY_OUTPUT_PATH/$NAME".
