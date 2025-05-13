# Setting up cBottle on NERSC (Perlmutter)

## Using containers
We can use containers on Perlmutter using [Shifter](https://docs.nersc.gov/development/containers/shifter/).

Login to the NERSC container registry using:
```bash
shifterimg login registry.nersc.gov
```

Pull the cbottle container using 
```bash
shifterimg pull registry.nersc.gov/m4935/cbottle
```

Launch a coarse res generation batch job using the submission script:
```bash
sbatch -A <ACCOUNT> scripts/nersc/submit_coarse_inference.sh
```
