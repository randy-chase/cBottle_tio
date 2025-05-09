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
from cbottle.diagnostics import visualize
import matplotlib.pyplot as plt


def test_visualize(tmp_path):
    x = torch.arange(12 * 8 * 8).float()
    visualize(x)
    p = tmp_path / "a.png"
    plt.savefig(p.as_posix())
    print(p)
    # to view plot
    # assert False
    # and open the printed path
