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

from cbottle.config.training.masking import MaskingConfig, StrategyWeights
from cbottle.training.video.frame_masker import FrameMasker


def sample_data(B=1, C=3, T=8, X=12):
    if B > 0:
        return {
            "target": torch.randn(B, C, T, X),
            "condition": torch.randn(B, 1, T, X),
        }
    else:
        return {
            "target": torch.randn(C, T, X),
            "condition": torch.randn(1, T, X),
        }


def test_frame_masker_random():
    B, C, T, X = 2, 3, 8, 12
    masker = FrameMasker(
        MaskingConfig(
            strategy_weights=StrategyWeights(
                random=1.0,
                blockwise=0.0,
                interpolation=0.0,
                full_dropout=0.0,
            ),
            random_mask_prob=0.8,
        )
    )
    out = masker(sample_data(B, C, T, X))
    assert "mask" in out
    assert out["mask"].shape == (B, 1, T, X)
    assert out["condition"].shape == (B, C + 1 + 1, T, X)

    # Check mask is identical across spatial dimension (X)
    mask_ref = out["mask"][..., 0]  # Reference slice [B, 1, T]
    assert torch.all(out["mask"] == mask_ref[..., None])  # Compare to all X positions

    # Some frames should be masked
    assert torch.any(~out["mask"])
    assert torch.any(out["mask"])


def test_frame_masker_blockwise():
    B, C, T, X = 2, 3, 8, 12
    config = MaskingConfig(
        strategy_weights=StrategyWeights(
            random=0.0,
            blockwise=1.0,
            interpolation=0.0,
            full_dropout=0.0,
        ),
        block_mask_fraction=0.8,
    )
    masker = FrameMasker(config)
    out = masker(sample_data(B, C, T, X))

    # Should have at most one change from masked to unmasked
    mask = out["mask"][0, 0, :, 0]
    changes = torch.where(mask[:-1] != mask[1:])[0]
    assert len(changes) <= 1


def test_frame_masker_endpoints():
    B, C, T, X = 2, 3, 8, 12
    config = MaskingConfig(
        strategy_weights=StrategyWeights(
            random=0.0,
            blockwise=0.0,
            interpolation=1.0,
            full_dropout=0.0,
        ),
    )
    masker = FrameMasker(config)
    out = masker(sample_data(B, C, T, X))
    mask = out["mask"]
    assert torch.all(mask[:, :, 0] == 1)
    assert torch.all(mask[:, :, -1] == 1)
    assert torch.all(mask[:, :, 1:-1] == 0)


def test_frame_masker_full_dropout():
    B, C, T, X = 2, 3, 8, 12
    config = MaskingConfig(
        strategy_weights=StrategyWeights(
            random=0.0,
            blockwise=0.0,
            interpolation=0.0,
            full_dropout=1.0,
        ),
    )
    masker = FrameMasker(config)
    out = masker(sample_data(B, C, T, X))
    assert torch.all(out["mask"] == 0)


def test_frame_masker_keep_frames():
    B, C, T, X = 2, 3, 8, 12
    keep_frames = [0, 4, 7]
    masker = FrameMasker(keep_frames=keep_frames)
    out = masker(sample_data(B, C, T, X))
    mask = out["mask"][0, 0, :, 0]
    assert torch.all(mask[keep_frames] == 1)
    assert torch.all(mask[list(set(range(T)) - set(keep_frames))] == 0)


def test_frame_masker_unbatched():
    C, T, X = 3, 8, 12
    config = MaskingConfig(
        strategy_weights=StrategyWeights(
            random=1.0,
            blockwise=0.0,
            interpolation=0.0,
            full_dropout=0.0,
        ),
    )
    masker = FrameMasker(config)
    unbatched = sample_data(B=-1, C=C, T=T, X=X)
    out = masker(unbatched)

    assert out["mask"].shape == (1, T, X)
    assert out["condition"].shape == (C + 1 + 1, T, X)
