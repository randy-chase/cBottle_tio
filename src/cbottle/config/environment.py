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
import os
import dotenv

dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))

CACHE_DIR = os.path.expanduser("~/.cache/cbottle")

#############
# ERA5 inputs
#############
V6_ERA5_ZARR = os.getenv(
    "V6_ERA5_ZARR", "/global/cfs/cdirs/trn006/data/nvidia/era5_hpx_6.zarr/"
)
V6_ERA5_ZARR_PROFILE = os.getenv("V6_ERA5_ZARR_PROFILE", "")

# SST data from CMIP
# Data is downloaded from the https://aims2.llnl.gov/search
AMIP_MID_MONTH_SST = os.getenv(
    "AMIP_MID_MONTH_SST",
    os.path.join(
        CACHE_DIR,
        "tosbcs_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-9_gn_187001-202212.nc",
    ),
)
AMIP_MID_MONTH_SST_PROFILE = os.getenv("AMIP_MID_MONTH_SST_PROFILE", "")

######
# ICON
######
RAW_DATA_URL = "/global/cfs/cdirs/trn006/data/nvidia/ngc3028_PT30M_4weeks_10.zarr/"
RAW_DATA_PROFILE = os.getenv("RAW_DATA_PROFILE", "")
RAW_DATA_URL_6 = os.getenv(
    "RAW_DATA_URL_6", "/global/cfs/cdirs/trn006/data/nvidia/ngc3028_PT30M_6.zarr/"
)

V6_ICON_ZARR = os.getenv(
    "V6_ICON_ZARR", "/global/cfs/cdirs/trn006/data/nvidia/ICON_v6_dataset.zarr/"
)

V6_ICON_ZARR_PROFILE = os.getenv("V6_ICON_ZARR_PROFILE", "")

LAND_DATA_URL_10 = os.getenv(
    "LAND_DATA_URL_10",
    "/global/cfs/cdirs/trn006/data/nvidia/landfraction/ngc3028_P1D_10.zarr/",
)
LAND_DATA_URL_6 = os.getenv(
    "LAND_DATA_URL_6",
    "/global/cfs/cdirs/trn006/data/nvidia/landfraction/ngc3028_P1D_6.zarr/",
)
LAND_DATA_PROFILE = os.getenv("LAND_DATA_PROFILE", "")


SST_MONMEAN_DATA_URL_6 = os.getenv(
    "SST_MONMEAN_DATA_URL_6",
    "/global/cfs/cdirs/trn006/data/nvidia/ngc3028_P1D_ts_monmean_6.zarr",
)
SST_MONMEAN_DATA_PROFILE = os.getenv("SST_MONMEAN_DATA_PROFILE", "")


# project file
PROJECT_ROOT = os.getenv(
    "PROJECT_ROOT", "/global/cfs/cdirs/trn006/data/nvidia/cBottle/"
)
DATA_ROOT = os.getenv("DATA_ROOT", os.path.join(PROJECT_ROOT, "datasets"))
CHECKPOINT_ROOT = os.getenv(
    "CHECKPOINT_ROOT", os.path.join(PROJECT_ROOT, "training-runs")
)
