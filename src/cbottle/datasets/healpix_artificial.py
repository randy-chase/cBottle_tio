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
import numpy as np
import torch
from cbottle import domain
from cbottle.datasets.base import SpatioTemporalDataset
from earth2grid import healpix


class HealPIXData(torch.utils.data.Dataset, SpatioTemporalDataset):
    """An artificial datasets consisting of sin/cosines defined on the sphere"""

    def __init__(self, level=6) -> None:
        self.level = level
        self._grid = healpix.Grid(level, pixel_order=healpix.HEALPIX_PAD_XY)

    @property
    def domain(self):
        return domain.HealPixDomain(self._grid)

    @property
    def num_channels(self) -> int:
        return 1

    @property
    def label_dim(self) -> int:
        return 0

    @property
    def time_length(self) -> int:
        return 1

    @property
    def condition_channels(self) -> int:
        # pure generation for now
        return 0

    def get(self, i):
        lat = self._grid.lat
        lon = self._grid.lon
        rng = np.random.default_rng(i)
        x = rng.integers(0, 6, size=(1,))
        y = rng.integers(0, 12, size=(1,))
        phi = rng.uniform(high=360)
        z = np.sin((2 * x) * np.deg2rad(lat)) * np.sin(y * np.deg2rad(lon - phi))
        # add singleton channel and time dimensions
        z_reshaped = z[None, None]
        label = np.empty([])
        n = np.prod(self._grid.shape)
        condition = np.zeros([self.condition_channels, self.time_length, n])
        return z_reshaped, label, condition

    def __getitem__(self, i):
        return self.get(i)

    def __len__(self):
        # return a massive number to approximate that this dataset is infinite
        # It doesn't seem that pytorch supports infinite sized map style datasets
        return 2**40
