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
from cbottle.datasets import dataset_3d
from cbottle.datasets.dataset_3d import INDEX

nchannel = len(INDEX)


def test_get_std():
    scale = dataset_3d.get_std()
    assert scale.shape == (nchannel,)


def test_get_mean():
    x = dataset_3d.get_mean()
    assert x.shape == (nchannel,)
