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
from cbottle.likelihood import log_prob
import torch
import math
import pytest


def D(x, t):
    # the denoiser for a normal gaussian x(0)
    # dx = - t s(x,t)
    # score = -x / (1 + t ** 2)
    # D =  t ** 2 * score + x = t ** 2 (-x) / (1 + t ** 2) + x
    return x / (1 + t**2)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("masked", [True, False])
def test_log_prob(masked, device):
    # Create an image tensor following a standard normal distribution
    torch.manual_seed(0)
    image = torch.randn((1, 256, 32)).to(device)
    if masked:
        image[0, :10] = torch.nan
        mask = ~torch.isnan(image)
        n = mask.sum().item()
    else:
        mask = None
        n = image.numel()

    # Call the logpdf function
    log_likelihood, soln = log_prob(
        D, image, mask=mask, divergence_samples=5, rtol=1e-4
    )
    assert soln.y[:, -1].std().item() == pytest.approx(soln.t[-1], rel=0.1)

    # E log p = - H(p)  =
    expected = -(math.log(2 * math.pi) + 1) / 2
    assert log_likelihood.cpu().numpy() / n == pytest.approx(expected, rel=0.01)


def test_log_prob_batch():
    # Create an image tensor following a standard normal distribution
    torch.manual_seed(0)
    image = torch.randn((2, 256, 32))
    image[1] = 0

    log_likelihood, soln = log_prob(D, image, divergence_samples=5, rtol=1e-4)

    # check the first batch
    n = image[0].numel()
    expected = -(math.log(2 * math.pi) + 1) / 2
    assert log_likelihood[0] / n == pytest.approx(expected, rel=0.01)

    # check the second sample
    expected = -math.log(2 * math.pi) / 2
    assert log_likelihood[1] / n == pytest.approx(expected, rel=0.01)
