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
#!/usr/bin/env python3
from earth2grid import healpix
import numpy as np
import matplotlib.pyplot as plt
import torch
import xarray as xr
import matplotlib.animation as animation
import matplotlib.colors
from dataclasses import dataclass
from typing import Annotated
from cbottle.dataclass_parser import parse_args, Help
import pandas as pd


@dataclass
class VideoOptions:
    input: Annotated[str, Help("Input netCDF file (.nc)")]
    output: Annotated[str, Help("Output video file (.mp4)")]
    style: Annotated[str, Help("Matplotlib style")] = "dark_background"
    fps: Annotated[int, Help("Frames per second")] = 10
    field: str = ""
    dpi: Annotated[int, Help("DPI for video quality")] = 100


def load_data(input_file: str) -> dict:
    ds = xr.open_dataset(input_file)
    data_vars = {}

    for field in ds.data_vars:
        mapping_var = ds[field].attrs.get("grid_mapping", None)
        if not mapping_var:
            continue
        if not ds[mapping_var].attrs["grid_mapping_name"] == "healpix":
            continue

        data = ds[field].values
        if ds[mapping_var].attrs["healpix_order"] == "nest":
            data = healpix.reorder(
                torch.as_tensor(data), healpix.PixelOrder.NEST, healpix.PixelOrder.RING
            ).numpy()
        data_vars[field] = data

    return data_vars, ds.time.values


def get_field_ranges() -> dict:
    return {
        "clivi": (-1e-12, 1.5),
        "cllvi": (0, 11),
        "hfls": (-1500, 400),
        "hfss": (-1700, 1600),
        "hydro_canopy_cond_limited_box": (0, 0.04),
        "hydro_discharge_ocean_box": (-800, 50000),
        "hydro_drainage_box": (0, 0.06),
        "hydro_runoff_box": (-0.0004, 0.03),
        "hydro_snow_soil_dens_box": (0, 300),
        "hydro_transpiration_box": (-0.0003, 0.00003),
        "hydro_w_snow_box": (0, 160),
        "pr": (0, 0.07),
        "pres_msl": (95000, 104000),
        "pres_sfc": (39000, 107000),
        "prls": (0, 0.008),
        "prw": (0, 70),
        "qgvi": (-1e-12, 31),
        "qrvi": (-1e-13, 40),
        "qsvi": (-1e-12, 15),
        "rlds": (70, 500),
        "rlus": (110, 800),
        "rlut": (70, 360),
        "rsds": (0, 1130),
        "rsdt": (0, 1380),
        "rsus": (0, 620),
        "rsut": (0, 1010),
        "sfcwind": (0, 45),
        "sic": (0, 1),
        "sit": (0, 12),
        "sse_grnd_hflx_old_box": (-280, 120),
        "tauu": (-10, 10),
        "tauv": (-11, 10),
        "uas": (-40, 40),
        "vas": (-40, 40),
        "speed": (0, 30),
    }


def process_field(data: np.ndarray, field: str) -> tuple:
    vmin, vmax = get_field_ranges().get(field, (data.min(), data.max()))

    if field == "pr":
        data = 10 * np.log10((data * 3600).clip(0.1))
        vmin, vmax = -5, 20
        cmap = "gist_ncar"
    elif field == "tas":
        data = data - 273.15
        vmin, vmax = -40, 40
        cmap = "RdBu_r"
    else:
        cmap = "bone"

    return data, vmin, vmax, cmap


def create_frame(data: np.ndarray, field: str, background: str) -> np.ndarray:
    data, vmin, vmax, cmap = process_field(data, field)

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    x = healpix.to_double_pixelization(data, fill_value=float("nan"))
    x = np.ma.masked_invalid(x)

    # Convert to RGB
    out_rgb = (plt.cm.get_cmap(cmap)(norm(x)) * 255).astype(np.uint8)
    return out_rgb


def main():
    args = parse_args(VideoOptions)

    # Load data
    data_vars, times = load_data(args.input)

    # Add speed field if uas and vas are present
    if "uas" in data_vars and "vas" in data_vars:
        data_vars["speed"] = np.sqrt(data_vars["uas"] ** 2 + data_vars["vas"] ** 2)

    # Filter fields if specified
    data_var = data_vars[args.field]

    # Create figure and animation
    plt.style.use(args.style)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_position([0, 0, 1, 1])
    ax.axis("off")

    data, vmin, vmax, cmap = process_field(data_var[0], args.field)
    x = healpix.to_double_pixelization(data, fill_value=float("nan"))
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    img = ax.imshow(
        np.ma.masked_invalid(x), extent=[0, 2, 0, 1], cmap=plt.get_cmap(cmap), norm=norm
    )
    plt.colorbar(img, ax=ax, orientation="horizontal", shrink=0.5, pad=0.02)
    title = ax.set_title(f"{args.field} ")
    ax.axis("off")
    fig.tight_layout()

    def update(frame):
        # Get the first field for demonstration
        field = args.field
        data = data_var[frame]
        data, vmin, vmax, cmap = process_field(data, field)
        x = healpix.to_double_pixelization(data, fill_value=float("nan"))
        x = np.ma.masked_invalid(x)
        img.set_data(x)
        time = pd.Timestamp(times[frame]).isoformat()
        title.set_text(f"{args.field} {time}")
        return [img, title]

    # Create animation
    n_frames = len(list(data_vars.values())[0])
    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 / args.fps, blit=True
    )

    # Save video
    writer = animation.FFMpegWriter(fps=args.fps)
    anim.save(args.output, writer=writer, dpi=args.dpi)
    plt.close()


if __name__ == "__main__":
    main()
