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
from datetime import datetime
from earth2studio.data import CBottle3D
from earth2studio.models.dx import CBottleSR


def test_e2s_cbottle_coarse(regtest):
    package = CBottle3D.load_default_package()
    ds = CBottle3D.load_model(package).to("cuda")
    cbottle_da = ds([datetime(2022, 9, 5)], ["msl", "tcwv"])
    cbottle_da.to_dataset(name="sample").info(regtest)


def test_e2s_cbottle_super_load():
    package = CBottleSR.load_default_package()
    super_resolution_window = (
        0,
        -120,
        50,
        -40,
    )  # (lat south, lon west, lat north, lon east)
    CBottleSR.load_model(
        package,
        output_resolution=(1024, 1024),
        super_resolution_window=super_resolution_window,
    )
