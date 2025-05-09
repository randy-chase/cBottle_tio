# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
from dataclasses import dataclass
from enum import Enum, auto

import cbottle.distributed as dist
import earth2grid
import torch
import torch.distributed
import tqdm
from cbottle.dataclass_parser import Help, a, parse_args
from cbottle.datasets import dataset_3d, samplers
from cbottle.datasets.dataset_2d import HealpixDatasetV5
from cbottle.netcdf_writer import NetCDFConfig, NetCDFWriter
from cbottle.denoiser_factories import get_denoiser, DenoiserType
from cbottle.datasets.base import BatchInfo

logger = logging.getLogger(__name__)


class Sampler(Enum):
    all = auto()
    fibonacci = auto()
    random = auto()


class Dataset(Enum):
    icon = auto()
    era5 = auto()
    amip = auto()


@dataclass(frozen=True)
class SamplerArgs:
    test_fast: a[bool, Help("Run in fast test mode")] = False
    regression: a[bool, Help("Run regression tests")] = False
    validate_only: a[bool, Help("Only perform validation")] = False
    min_samples: a[int, Help("Minimum number of samples.")] = -1
    start_from_noisy_image: a[bool, Help("Start from a noisy image")] = False
    sigma_max: a[float, Help("Maximum sigma value")] = 80.0
    sampler: Sampler = Sampler.all

    denoiser_when_nan: a[str, Help("Choices: ['', 'icon']")] = ""
    denoiser_type: a[
        DenoiserType, Help("Choices: ['mask_filling', 'infill', 'standard']")
    ] = DenoiserType.standard
    save_data: bool = False
    bias_correct: bool = False
    """If True, then bias correct the input data to look like ERA5
    """
    bf16: a[bool, Help("Use bf16")] = False
    batch_gpu: a[int, Help("Batch size per GPU")] = 4
    seed: int | None = None


@dataclass
class CLI:
    state_path: a[str, Help("Path to the model state file")]
    output_path: a[str, Help("Path to the output directory")]
    dataset: Dataset = Dataset.icon
    data_split: str = ""
    sst_offset: float = 0.0
    sample: SamplerArgs = SamplerArgs()
    hpx_level: int = 6


units = "seconds since 1900-1-1 0:0:0"
calendar = "proleptic_gregorian"


def build_labels(labels, denoiser_when_nan: str):
    out_labels = torch.zeros_like(labels)
    if denoiser_when_nan == "icon":
        out_labels[:, HealpixDatasetV5.LABEL] = 1
    return out_labels


def prepare_for_saving(
    x: torch.Tensor, hpx: earth2grid.healpix.Grid, batch_info: BatchInfo
):
    """
    Denormalizes and reorders data to RING order
    """
    x = batch_info.denormalize(x)
    ring_order = hpx.reorder(earth2grid.healpix.PixelOrder.RING, x)
    return {batch_info.channels[c]: ring_order[:, c] for c in range(x.shape[1])}


def save_inferences(
    net,
    dataset,
    output_path,
    *,
    attrs=None,
    hpx_level: int,
    config: SamplerArgs,
    rank: int,
    world_size: int,
):
    attrs = attrs or {}
    tasks = None

    # Initialize netCDF writer
    nc_config = NetCDFConfig(
        hpx_level=hpx_level,
        time_units=dataset.time_units,
        calendar=dataset.calendar,
        attrs=attrs,
    )
    writer = NetCDFWriter(
        output_path, nc_config, dataset.batch_info.channels, rank=rank
    )

    if config.sampler == Sampler.fibonacci:
        sampler = samplers.subsample(dataset, min_samples=config.min_samples)
        tasks = samplers.distributed_split(sampler)
    elif config.sampler == Sampler.all:
        if config.min_samples > 0:
            dataset.set_times(dataset.times[: config.min_samples])

        if hasattr(dataset, "infinite"):
            dataset.infinite = False
        if hasattr(dataset, "shuffle"):
            dataset.shuffle = False

        # Skip times that have already been processed
        try:
            logger.info(
                f"Skipping {writer.time_index} times out of {len(dataset._times)}"
            )
            dataset._times = dataset._times[writer.time_index :]
        except AttributeError:
            pass

    loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=config.batch_gpu, sampler=tasks, drop_last=False
    )

    for batch in tqdm.tqdm(loader, disable=rank != 0):
        if (config.min_samples > 0) and (
            writer.time_index * world_size > config.min_samples
        ):
            break
        images, labels, condition = batch["target"], batch["labels"], batch["condition"]
        second_of_day = batch["second_of_day"].cuda().float()
        day_of_year = batch["day_of_year"].cuda().float()
        # TODO parameterize
        with torch.no_grad():
            condition = condition.cuda()
            labels = labels.cuda()
            images = images.cuda()

            from cbottle.diffusion_samplers import (
                StackedRandomGenerator,
                edm_sampler_from_sigma,
            )

            device = condition.device

            if config.seed is None:
                rnd = torch
            else:
                rnd = StackedRandomGenerator(
                    device, seeds=[config.seed] * images.shape[0]
                )

            latents = rnd.randn(
                (
                    images.shape[0],
                    net.img_channels,
                    net.time_length,
                    net.domain.numel(),
                ),
                device=device,
            )

            if config.start_from_noisy_image:
                xT = latents * config.sigma_max + images
            else:
                xT = latents * config.sigma_max

            labels_when_nan = None
            if config.denoiser_type == DenoiserType.mask_filling:
                labels_when_nan = build_labels(labels, config.denoiser_when_nan)
            elif config.denoiser_type == DenoiserType.infill:
                labels = build_labels(labels, config.denoiser_when_nan)

            # Gets appropriate denoiser based on config
            D = get_denoiser(
                net=net,
                images=images,
                labels=labels,
                condition=condition,
                second_of_day=second_of_day,
                day_of_year=day_of_year,
                denoiser_type=config.denoiser_type,
                sigma_max=config.sigma_max,
                labels_when_nan=labels_when_nan,
            )

            if config.save_data:
                out = images
            elif config.bias_correct:
                # First encode with noise
                tmin = 0.02
                tmax = int(config.sigma_max)  # Convert to int for type compatibility
                y0 = images + torch.randn_like(images) * tmin

                encoded = edm_sampler_from_sigma(
                    D,
                    y0,
                    sigma_max=tmax,
                    sigma_min=tmin,
                    num_steps=24,
                    randn_like=torch.randn_like,
                    reverse=True,
                    S_noise=0,
                )

                labels_when_nan = torch.zeros_like(batch["labels"].cuda())
                labels_when_nan[:, 0] = 1.0
                era5labels = torch.nn.functional.one_hot(
                    torch.tensor([1], device=condition.device), 1024
                )

                denoiser_era5 = get_denoiser(
                    net=net,
                    images=images,
                    labels=era5labels,
                    condition=condition,
                    second_of_day=second_of_day,
                    day_of_year=day_of_year,
                    denoiser_type=DenoiserType.mask_filling,
                    labels_when_nan=labels_when_nan,
                )

                # Then decode with ERA5 labels
                out = edm_sampler_from_sigma(
                    denoiser_era5,
                    encoded,
                    sigma_max=tmax,
                    sigma_min=tmin,
                    randn_like=torch.randn_like,
                    num_steps=24,
                    S_noise=0,
                )
            else:
                with torch.autocast("cuda", enabled=config.bf16, dtype=torch.bfloat16):
                    out = edm_sampler_from_sigma(
                        D,
                        xT,
                        randn_like=torch.randn_like,
                        sigma_max=int(
                            config.sigma_max
                        ),  # Convert to int for type compatibility
                    )

            ring_denormalized_data = prepare_for_saving(
                out, net.domain._grid, dataset.batch_info
            )

            # Convert time data to timestamps
            timestamps = batch["timestamp"]

            writer.write_batch(ring_denormalized_data, timestamps)


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args(CLI, convert_underscore_to_hyphen=False)
    state_path = args.state_path

    dist.init()

    # get dataset options from loop
    from cbottle.checkpointing import Checkpoint

    with Checkpoint(state_path) as checkpoint:
        net = checkpoint.read_model()

    net.eval()
    net.requires_grad_(False)
    net.float()
    net.cuda()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # get slurm vars if present
    id = int(os.getenv("SLURM_ARRAY_TASK_ID", "1"))  # 1-indexed
    slurm_array_count = int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))

    rank = rank + world_size * (id - 1)
    world_size = world_size * slurm_array_count

    # only one time step in this:
    # dataset = Era5Dataset(args.hpx_level, train=False)
    dataset = dataset_3d.get_dataset(
        rank=rank,
        world_size=world_size,
        split=args.data_split,
        dataset=args.dataset.name,
        sst_input=True,
        infinite=False,
        shuffle=False,
    )

    dataset.infinite = False

    save_inferences(
        net,
        dataset,
        args.output_path,
        hpx_level=args.hpx_level,
        config=args.sample,
        rank=rank,
        world_size=world_size,
    )


if __name__ == "__main__":
    main()
