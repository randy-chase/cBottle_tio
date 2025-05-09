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
"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
import scipy.stats
import numpy as np
from typing import Literal


# ----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).
class EDMLoss:
    def __init__(
        self,
        P_mean=-1.2,
        P_std=1.2,
        sigma_data=0.5,
        distribution: Literal["log_normal", "log_uniform", "power"] = "log_normal",
        sigma_max=1000.0,
        sigma_min=0.02,
    ):
        self.distribution = distribution
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def _sample_sigma_like_v1(self, x):
        rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        return sigma

    def _sample_sigma_like_loguniform(self, x):
        sigma_min = torch.tensor(self.sigma_min, device=x.device)
        sigma_max = torch.tensor(self.sigma_max, device=x.device)
        u = torch.rand([x.shape[0], 1, 1, 1], device=x.device)
        us = torch.log(sigma_min) + u * (torch.log(sigma_max) - torch.log(sigma_min))
        return torch.exp(us)

    def _sample_sigma_like_power(self, x):
        rho = 7
        sigma_min = torch.tensor(self.sigma_min, device=x.device)
        sigma_max = torch.tensor(self.sigma_max, device=x.device)
        u = torch.rand([x.shape[0], 1, 1, 1], device=x.device)

        def f(x):
            return x**rho

        def fi(t):
            return t ** (1 / rho)

        us = (1 - u) * fi(sigma_min) + u * fi(sigma_max)
        return f(us)

    def get_edm_pdf(self, sigma):
        # rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        # sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        # log(sigma) = Z * P_std + P_mean
        # (log(sigma) - P_mean)/P_std = Z
        # int p_sigma(s) ds = int p_z(z) dz = int p_z(z(s)) dz/ds ds = 1
        z = (np.log(sigma) - self.P_mean) / self.P_std
        dz_ds = 1 / sigma / self.P_std
        p_z = scipy.stats.norm.pdf(z)
        return p_z * dz_ds

    def pdf(self, sigma):
        if self.distribution == "log_normal":
            return self.get_edm_pdf(sigma)
        elif self.distribution == "log_uniform":
            p_log_s = 1 / (np.log(self.sigma_max) - np.log(self.sigma_min))
            p_log_s = np.where(
                (sigma >= self.sigma_min) & (sigma <= self.sigma_max), p_log_s, 0
            )
            return p_log_s / sigma

    def __call__(self, net, images):
        if self.distribution == "log_normal":
            sigma = self._sample_sigma_like_v1(images)
        elif self.distribution == "log_uniform":
            sigma = self._sample_sigma_like_loguniform(images)
        elif self.distribution == "power":
            sigma = self._sample_sigma_like_power(images)
        else:
            raise NotImplementedError(self.distribution)

        _, _, _t, _x = range(4)
        mask = ~torch.isnan(images).any(dim=(_t, _x), keepdim=True)
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        y = images
        n = torch.randn_like(y) * sigma
        D_yn = net(torch.where(mask, y + n, 0), sigma)
        loss = weight * ((D_yn - torch.where(mask, y, 0)) ** 2)
        loss = torch.where(mask, loss, 0) / mask.float().mean()
        return loss


class RegressLoss:
    SIGMA = 10000

    def __init__(
        self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, distribution=1, sigma_max=1000.0
    ):
        self.distribution = distribution
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, condition=None, labels=None, augment_labels=None):
        # for compatibility with edmprecond use a large SIGMA value must be the
        # same at inference time.

        # this large sigma ensures: weight=1.0, c_out=1.0, c_in=0.0, c_skip=0.0
        sigma = (self.SIGMA) * torch.ones(
            [images.shape[0], 1, 1, 1], device=images.device
        )
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        y = images
        # for regression sigma_data=0, makes sure c_out=1, c_skip=0 in the EDMprecond
        D_yn = net(
            torch.zeros_like(y),
            sigma,
            labels,
            augment_labels=augment_labels,
            condition=condition,
        )
        loss = weight * ((D_yn - y) ** 2)
        return loss
