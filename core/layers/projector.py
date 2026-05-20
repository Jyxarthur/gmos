# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict, Any

logger = logging.getLogger(__name__)


class TokenFuser(nn.Module):
    def __init__(self, c1: int, c2: int, embed_dim: int, hidden: int = 512, p_drop: float = 0.):
        super().__init__()
        self.proj1 = nn.Linear(c1, hidden, bias=True)
        self.proj2 = nn.Linear(c2, hidden, bias=True)
        self.fuse  = nn.Linear(2 * hidden, embed_dim, bias=True)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(p_drop)
        self.norm  = nn.LayerNorm(embed_dim, eps=1e-6)

        self.apply(lambda m: nn.init.trunc_normal_(m.weight, std=0.02) if isinstance(m, nn.Linear) else None)

    def forward(self, t1, t2):            # both [B, N, Ci]
        x = torch.cat([self.proj1(t1), self.proj2(t2)], dim=-1)  # [B, N, 2*hidden]
        x = self.act(self.fuse(x))         # [B, N, embed_dim]
        x = self.drop(x)
        return self.norm(x)


class TokenProjector(nn.Module):
    def __init__(self, c1: int, embed_dim: int, p_drop: float = 0.):
        super().__init__()
        self.proj = nn.Linear(c1, embed_dim, bias=True)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(p_drop)
        self.norm  = nn.LayerNorm(embed_dim, eps=1e-6)

        self.apply(lambda m: nn.init.trunc_normal_(m.weight, std=0.02) if isinstance(m, nn.Linear) else None)

    def forward(self, t1):            # both [B, N, Ci]
        x = self.proj(t1)       # [B, N, 2*hidden]
        x = self.act(x)         # [B, N, embed_dim]
        x = self.drop(x)
        return self.norm(x)