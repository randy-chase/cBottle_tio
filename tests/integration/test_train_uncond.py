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
import cbottle.distributed as dist
import pytest
import torch
import train_coarse
from cbottle.storage import StorageConfigError


def test_train_uncond(tmpdir):
    if not torch.distributed.is_initialized():
        dist.init()

    loop = train_coarse.TrainingLoop(
        run_dir=str(tmpdir),
        batch_size=2,
        batch_gpu=2,
        steps_per_tick=1,
        total_ticks=1,
        valid_min_samples=1,
        dataloader_num_workers=2,
        hpx_level=6,
        with_era5=False,
    )

    try:
        loop.setup()
        loop.train()
    except StorageConfigError:
        pytest.skip()
