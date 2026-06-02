"""IdentityEncoder: asparagus-compatible wrapper for precomputed embeddings.

asparagus LinearProbeModule._get_features() contract:
    skips = model._encode(x)          # x: (B, ...)
    deepest = skips[-1]               # shape (B, C, D, H, W)
    features = global_pool(deepest)   # AdaptiveAvgPool3d → (B, C, 1, 1, 1)
    return flatten(features, 1)       # → (B, C)

IdentityEncoder reshapes a flat embedding (B, feature_dim) to
(B, feature_dim, 1, 1, 1) so that the subsequent pool + flatten is a no-op.

LinearProbeModule.__init__ also reads model.decoder.fc.in_features to size
the linear head.  _FakeDecoder provides that attribute.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _FakeFC:
    def __init__(self, in_features: int) -> None:
        self.in_features = in_features


class _FakeDecoder:
    def __init__(self, in_features: int) -> None:
        self.fc = _FakeFC(in_features)


class IdentityEncoder(nn.Module):
    """Pass-through encoder for precomputed 1D embeddings.

    Input  x : (batch, feature_dim)
    Output   : list containing (batch, feature_dim, 1, 1, 1)

    The returned tensor satisfies the asparagus _get_features() contract:
    AdaptiveAvgPool3d((1,1,1)) is a no-op on (*, 1, 1, 1), and
    flatten(1) produces (batch, feature_dim) as expected by linear heads.
    """

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self._feature_dim = feature_dim
        self.decoder = _FakeDecoder(feature_dim)

    def _encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [x.view(x.shape[0], self._feature_dim, 1, 1, 1)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def parameters(self, recurse=True):
        return iter([])
