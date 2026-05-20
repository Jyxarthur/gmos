# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Type

import torch
from torch import nn
import torch.nn.functional as F

from .basic import LayerNorm2d, MLP


class UpSampling(nn.Module):
    def __init__(
        self,
        transformer_dim: int,
        num_objs: int = 10,
        activation: Type[nn.Module] = nn.GELU,
        use_high_res_features: bool = False,
        **kwargs,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.use_high_res_features = use_high_res_features

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )


        self.output_hypernetwork_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)


    def forward(
        self,
        src: torch.Tensor,
        obj_tokens: torch.Tensor,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        b, _, C = obj_tokens.shape

        
        hyper_in = self.output_hypernetwork_mlp(obj_tokens)
        
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)
            
        b, c, h, w = upscaled_embedding.shape

        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        return masks


class PointUpSampling(nn.Module):
    def __init__(
        self,
        transformer_dim: int,
        num_objs: int = 10,
        point_size: int = 64,
        **kwargs,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.point_size = point_size
        self.output_hypernetwork_mlp = MLP(transformer_dim, transformer_dim, transformer_dim, 3)

    def forward(
        self,
        src: torch.Tensor,
        obj_tokens: torch.Tensor,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        b, _, C = obj_tokens.shape
        hyper_in = self.output_hypernetwork_mlp(obj_tokens)
        resized_embedding = F.interpolate(src, size=(self.point_size, self.point_size), mode="bilinear", align_corners=False)

        b, c, h, w = resized_embedding.shape
        masks = (hyper_in @ resized_embedding.view(b, c, h * w)).view(b, -1, h, w)

        return masks