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
import argparse
import os
import sys
import xarray as xr
from pathlib import Path

AMIP_SST_FILENAME = (
    "tosbcs_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-9_gn_187001-202212.nc"
)
AMIP_SST_URL = (
    "https://esgf.ceda.ac.uk/thredds/dodsC/esg_cmip6/input4MIPs/CMIP6Plus/CMIP/"
    f"PCMDI/PCMDI-AMIP-1-1-9/ocean/mon/tosbcs/gn/v20230512/{AMIP_SST_FILENAME}"
)
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/cbottle")


def download_sst(output_path: str):
    try:
        print(f"Downloading SST data to {output_path} ...")
        ds = xr.open_dataset(AMIP_SST_URL)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        ds.to_netcdf(output_path)  # Save to netCDF
        print("Successfully downloaded SST data")

    except Exception as e:
        print(f"Error downloading SST data: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        help="Path to save the AMIP SST NetCDF file",
        default=os.path.join(DEFAULT_CACHE_DIR, AMIP_SST_FILENAME),
    )

    args = parser.parse_args()
    download_sst(args.output_path)


if __name__ == "__main__":
    main()
