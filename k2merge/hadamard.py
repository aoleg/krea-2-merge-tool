"""Block Hadamard rotation used by the ComfyUI int8 "convrot" format.

Vendored from comfy-kitchen 0.2.31, ``comfy_kitchen/tensor/int8_utils.py``
(functions ``_build_hadamard`` and ``_rotate_weight``).

Copyright (c) 2025 Comfy Org. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License"); you may not
use this file except in compliance with the License. You may obtain a copy of
the License at http://www.apache.org/licenses/LICENSE-2.0. Unless required by
applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied. See the License for the specific language
governing permissions and limitations under the License.

See THIRD_PARTY_NOTICES.md at the repository root.

The rotation matrix is a normalized regular Hadamard matrix. It is orthogonal
and symmetric, so ``rotate_weight`` applied twice is the identity: the same
function quantizes (rotate) and dequantizes (un-rotate).
"""
from __future__ import annotations

import math

import torch

_HADAMARD_CACHE: dict = {}


def build_hadamard(size: int, device="cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Build a normalized REGULAR orthogonal Hadamard matrix (ConvRot)."""
    device = torch.device(device)
    cache_key = (size, str(device), dtype)
    if cache_key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[cache_key]

    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")

    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=dtype,
        device=device,
    )

    h = h4
    current_size = 4
    while current_size < size:
        h = torch.kron(h, h4)
        current_size *= 4

    h_normalized = h / (size**0.5)
    _HADAMARD_CACHE[cache_key] = h_normalized
    return h_normalized


def rotate_weight(weight: torch.Tensor, h: torch.Tensor, group_size: int) -> torch.Tensor:
    """Rotate a 2-D weight offline: W_rot = W @ H_block^T, per group of input columns."""
    out_f, in_f = weight.shape
    if in_f % group_size != 0:
        raise ValueError(f"in_features {in_f} not divisible by group_size {group_size}")
    n_groups = in_f // group_size

    weight_grouped = weight.reshape(out_f, n_groups, group_size)
    h_t = h.T.to(dtype=weight.dtype, device=weight.device)
    weight_rotated = torch.matmul(weight_grouped, h_t)
    return weight_rotated.reshape(out_f, in_f)


def unrotate_weight(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    """Inverse of rotate_weight. H is symmetric and orthogonal, so this is rotate_weight again."""
    h = build_hadamard(group_size, device=weight.device, dtype=weight.dtype)
    return rotate_weight(weight, h, group_size)
