# Climate in a Bottle
![](https://private-user-images.githubusercontent.com/1386642/442340656-bbd375ef-47d1-466f-ad19-b5ea11376ef3.jpg?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDY4MzM3MjIsIm5iZiI6MTc0NjgzMzQyMiwicGF0aCI6Ii8xMzg2NjQyLzQ0MjM0MDY1Ni1iYmQzNzVlZi00N2QxLTQ2NmYtYWQxOS1iNWVhMTEzNzZlZjMuanBnP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI1MDUwOSUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNTA1MDlUMjMzMDIyWiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9Yjg5ODk0MjkzYmQzZjRmOWU2ZjdiNWFkYWI5OTAwMjM1YzliYTY5NjNjY2NkYTgxNDJkZDkwNjYxZjUwZWI3ZSZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QifQ.Puy94EkhszwwH2El4_w707ponMiii6uNi-XZWaZtyYU)

cBottle is an diffusion model that generates atmospheric states at kilometer resolution using a cascaded diffusion architecture.

This model is for research and development only.

## Setup

See [installation](docs/installation.md) instructions.

## Coarse Model (cBottle-3d)

### Training

```
python3 scripts/train_coarse.py --loop.noise_distribution log_uniform --loop.sigma_min 0.02 --loop.sigma_max 200 --loop.label_dropout 0.25 --loop.batch_gpu 4 --loop.batch_size 64 --loop.dataloader_num_workers 8 --loop.with_era5 --loop.use_labels  --loop.data_version 6 --loop.monthly_sst_input --name v6data  --loop.dataloader_prefetch_factor 100
```

### Inference

See `scripts/inference_coarse.py`.

## Coarse Video Model (cBottle-video)

###
Video training requires larger chunk sizes than image training.

```
python3 scripts/train_coarse.py
    --name v6-video \
    --loop.time_length 12 \
    --loop.time_step 6 \
    --loop.icon_chunk_size 56 \
    --loop.era5_chunk_size 96 \
    --loop.use_labels \
    --loop.label_dropout 0.05 \
    --loop.with_era5 \
    --loop.monthly_sst_input \
    --loop.noise_distribution log_uniform \
    --loop.sigma_min 0.02 \
    --loop.sigma_max 1000 \
    --loop.network.model_channels 256 \
    --loop.snapshot_ticks 1 \
    --loop.state_dump_ticks 1 \
    --loop.steps_per_tick 2000 \
    --loop.batch_gpu 1 \
    --loop.batch_size 32 \
    --loop.valid_min_samples 32 \
    --loop.dataloader_prefetch_factor 10
    --loop.dataloader_num_workers 8
```

### Inference
To create netcdf files of the generations (and optionally the corresponding ground truth), run the following:
```
torchrun --nproc-per-node 8 scripts/inference_coarse_video.py \
    /path/to/your/model.checkpoint \
    --output_path /output/path \
    --sample.save_mode all \
    --sample.frame_selection_strategy unconditional \
    --sample.denoiser_type standard \
    --sample.sigma_max 0.02 \
    --sample.sigma_max 1000
```

## Super-resolution model (cBottle-SR)

### Training
```
python3 scripts/train_multidiffusion.py --output-path OUTPUT 
```

### Inference

```
python3 scripts/inference_multidiffusion.py --input-path path/to/checkpoint output/
```


## Disclaimer

This project will download and install additional third-party open source
software projects. Review the license terms of these open source projects before
use.

