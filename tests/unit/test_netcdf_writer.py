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
import torch
from cbottle.netcdf_writer import NetCDFWriter, NetCDFConfig
import netCDF4 as nc
import numpy as np


def test_NetCDFWriter(tmp_path):
    # Setup
    output_path = tmp_path / "nc_out"
    config = NetCDFConfig(hpx_level=1)  # small nside for test
    channels = ["var1", "var2"]
    writer = NetCDFWriter(str(output_path), config, channels, rank=0)

    # Prepare data
    npix = 12 * (2**config.hpx_level) ** 2
    batch_size = 2
    output_data = {
        "var1": torch.ones((batch_size, 1, npix), dtype=torch.float32),
        "var2": torch.zeros((batch_size, 1, npix), dtype=torch.float32),
    }
    timestamps = torch.tensor([1000, 2000], dtype=torch.int64)
    writer.write_batch(output_data, timestamps)

    # Check file exists
    nc_path = output_path / "0.nc"
    assert nc_path.exists()

    # Check contents
    with nc.Dataset(nc_path, "r") as ds:
        assert set(ds.variables) >= {"var1", "var2", "time", "crs"}
        np.testing.assert_array_equal(ds["time"][:2], [1000, 2000])
        np.testing.assert_array_equal(ds["var1"][:2], np.ones((2, npix)))
        np.testing.assert_array_equal(ds["var2"][:2], np.zeros((2, npix)))

    # Test reinitialization - create new writer with same path
    writer2 = NetCDFWriter(str(output_path), config, channels, rank=0)

    # Write more data
    timestamps2 = torch.tensor([3000, 4000], dtype=torch.int64)
    writer2.write_batch(output_data, timestamps2)

    # Check appended contents
    with nc.Dataset(nc_path, "r") as ds:
        assert len(ds["time"]) == 4
        np.testing.assert_array_equal(ds["time"][:], [1000, 2000, 3000, 4000])
        np.testing.assert_array_equal(ds["var1"][:], np.ones((4, npix)))
        np.testing.assert_array_equal(ds["var2"][:], np.zeros((4, npix)))


def test_NetCDFWriter_video_mode(tmp_path):
    # Setup
    output_path = tmp_path / "nc_out_video"
    config = NetCDFConfig(hpx_level=1)  # small nside for test
    channels = ["var1"]
    writer = NetCDFWriter(
        str(output_path), config, channels, rank=0, add_video_variables=True
    )

    # Prepare data
    npix = 12 * (2**config.hpx_level) ** 2
    batch_size = 2
    time_length = 3
    output_data = {
        "var1": torch.ones((batch_size, time_length, npix), dtype=torch.float32),
    }
    timestamps = torch.tensor(
        [[1000, 2000, 3000], [1500, 2500, 3500]], dtype=torch.int64
    )

    # Add video-specific metadata
    video_metadata = {
        "frame_source_flag": torch.tensor([[1, 1, 1], [2, 2, 2]], dtype=torch.int8),
        "lead_time": torch.tensor([[0, 6, 12], [0, 6, 12]], dtype=torch.int32),
    }

    writer.write_batch(output_data, timestamps, scalars=video_metadata)

    # Check file exists
    nc_path = output_path / "0.nc"
    assert nc_path.exists()

    # Check contents
    with nc.Dataset(nc_path, "r") as ds:
        assert set(ds.variables) >= {
            "var1",
            "time",
            "crs",
            "frame_source_flag",
            "lead_time",
        }
        assert len(ds["time"]) == batch_size * time_length

        np.testing.assert_array_equal(ds["frame_source_flag"][:], [1, 1, 1, 2, 2, 2])
        np.testing.assert_array_equal(ds["lead_time"][:], [0, 6, 12, 0, 6, 12])

        np.testing.assert_array_equal(
            ds["time"][:], [1000, 2000, 3000, 1500, 2500, 3500]
        )
        np.testing.assert_array_equal(
            ds["var1"][:], np.ones((batch_size * time_length, npix))
        )
