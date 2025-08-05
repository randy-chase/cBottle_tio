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
from cbottle.inference import MixtureOfExpertsDenoiser
from cbottle.models.networks import Output
import cbottle.domain
import earth2grid
import torch


def test_moe():
    class Net(torch.nn.Module):
        def __init__(self, sigma_min, sigma_max):
            super().__init__()
            self.sigma_max = sigma_max
            self.sigma_min = sigma_min
            self.domain = cbottle.domain.HealPixDomain(
                earth2grid.healpix.Grid(6, earth2grid.healpix.HEALPIX_PAD_XY)
            )

        def round_sigma(self, sigma):
            return torch.as_tensor(sigma)

        def forward(self, x, t, **kwargs):
            assert t > self.sigma_min
            assert t < self.sigma_max
            return Output(x)

    x = torch.randn([4, 64, 64])
    d = MixtureOfExpertsDenoiser([Net(10.0, 100.0), Net(0, 10.0)], [10.0])
    for t in range(11, 0, -5):
        d(x, torch.tensor(t))  # will raise error if not done correctly.
