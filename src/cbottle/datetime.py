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
import datetime
import cftime


def as_pydatetime(time) -> datetime.datetime:
    if isinstance(time, cftime.datetime):
        # very important to set the timezone to UTC for example when using timestamps
        return datetime.datetime(*cftime.to_tuple(time), tzinfo=datetime.timezone.utc)
    elif isinstance(time, datetime.datetime):
        return time
    else:
        raise NotImplementedError(type(time))


def second_of_day(time):
    begin_of_day = time.replace(hour=0, second=0, minute=0)
    return (time - begin_of_day).total_seconds()


def as_cftime(timestamp):
    return cftime.DatetimeGregorian(
        timestamp.year,
        timestamp.month,
        timestamp.day,
        timestamp.hour,
        timestamp.minute,
        timestamp.second,
    )
