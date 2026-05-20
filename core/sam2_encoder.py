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

# from .position_encoding import PositionEmbeddingRandom
# from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter

# from .transformer import TransformerEncoder, TransformerDecoder, TwoSourceTransformerDecoder, TwoSourcePartialTwoWayTransformerDecoder
# from .token_fuser import TokenFuser, TokenProjector
# from .mask_upsampling import UpSampling
# from .model_utils import MLP
    
logger = logging.getLogger(__name__)


import torch.distributed
import torch.nn.functional as F

from torch.nn.init import trunc_normal_

from .sam2.sam2.modeling.sam.mask_decoder import MaskDecoder
from .sam2.sam2.modeling.sam.prompt_encoder import PromptEncoder
from .sam2.sam2.modeling.sam.transformer import TwoWayTransformer
from .sam2.sam2.modeling.sam2_utils import get_1d_sine_pe, MLP, select_closest_cond_frames

from .sam2.sam2.modeling.backbones.image_encoder import FpnNeck, ImageEncoder
from .sam2.sam2.modeling.backbones.hieradet import Hiera
from .sam2.sam2.modeling.position_encoding import PositionEmbeddingSine
# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0


class SkipConnect(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv_s0 = nn.Conv2d(hidden_dim, hidden_dim // 8, kernel_size=1, stride=1)
        self.conv_s1 = nn.Conv2d(hidden_dim, hidden_dim // 4, kernel_size=1, stride=1)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Please use the corresponding methods in SAM2VideoPredictor for inference or SAM2Train for training/fine-tuning"
            "See notebooks/video_predictor_example.ipynb for an inference example."
        )


class SAM2Encoder(nn.Module):
    def __init__(
        self,
        num_objs=10,
        backbone_stride=16,  # stride of the image backbone output
        image_size=1024,
        use_high_res_features_in_sam=True,  # whether to use high-resolution feature maps in the SAM mask decoder
        directly_add_no_mem_embed=True,
        compile_image_encoder: bool=False,
    ):
        super().__init__()

        self.num_objs = num_objs
        self._build_image_encoder()
        self.use_high_res_features_in_sam = use_high_res_features_in_sam
        self.num_feature_levels = 3 if use_high_res_features_in_sam else 1

        self.hidden_dim = self.image_encoder.neck.d_model

        self.image_size = image_size
        self.backbone_stride = backbone_stride

        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]

        self.sam_mask_decoder = SkipConnect(self.hidden_dim)    

        self.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.no_mem_embed, std=0.02)
        self.directly_add_no_mem_embed = directly_add_no_mem_embed      

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Please use the corresponding methods in SAM2VideoPredictor for inference or SAM2Train for training/fine-tuning"
            "See notebooks/video_predictor_example.ipynb for an inference example."
        )
    def _build_image_encoder(self):
        # === Instantiate position encoding ===
        position_encoding = PositionEmbeddingSine(
            num_pos_feats=256,
            normalize=True,
            scale=None,
            temperature=10000
        )

        # === Instantiate neck ===
        neck = FpnNeck(
            position_encoding=position_encoding,
            d_model=256,
            backbone_channel_list=[1152, 576, 288, 144],
            fpn_top_down_levels=[2, 3],
            fpn_interp_model="nearest"
        )

        # === Instantiate trunk ===
        trunk = Hiera(
            embed_dim=144,
            num_heads=2,
            stages=[2, 6, 36, 4],
            global_att_blocks=[23, 33, 43],
            window_pos_embed_bkg_spatial_size=[7, 7],
            window_spec=[8, 4, 16, 8]
        )

        # === Instantiate image encoder ===
        self.image_encoder = ImageEncoder(
            scalp=1,
            trunk=trunk,
            neck=neck
        )


    def forward_image(self, img_batch: torch.Tensor):
        """Get the image feature on the input batch."""
        backbone_out = self.image_encoder(img_batch)
        if self.use_high_res_features_in_sam:
            # precompute projected level 0 and level 1 features in SAM decoder
            # to avoid running it again on every SAM click
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0]
            )
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1]
            )
        return backbone_out

    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def _get_vision_features(self, vision_feats):
        _, bs, _ = vision_feats[0].shape

        if self.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.no_mem_embed
        
        feats = [
            feat.permute(1, 2, 0).view(bs, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]

        return feats[-1], feats[:-1]
