"""GMOS proposer: the multi-object query-based segmentation model.

Fuses geometric features from the frozen Pi3 encoder with segmentation
features from the frozen SAM2 image encoder at a shared 64×64 grid, refines
them with a self-attention encoder, and decodes N object queries via a
two-way transformer. Each query yields a mask, a motion-state prediction,
a mask-IoU estimate, and a confidence score.
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.transformer import TransformerEncoder, TwoWayTransformer
from .layers.rope import RotaryPositionEmbedding2D, PositionGetter
from .layers.basic import MLP
from .layers.upsampling import UpSampling
from .layers.projector import TokenProjector

logger = logging.getLogger(__name__)


class GMOSProposer(nn.Module):
    """GMOS proposer.

    Pi3 and SAM2 features are projected, resized to a shared 64×64 grid,
    concatenated, and merged before going through a transformer encoder
    (self-attention) and a two-way decoder that attends N object queries
    to the merged features. Each query yields a mask, motion state, mask-IoU
    estimate, and confidence score.

        pi3_feat (B,h,w,1024) -> proj(1024->256) -> resize 64x64  \\
                                                                     concat(512) -> merge_proj(512->256)
        sam_feat (B,256,64,64)             -> proj(256->256)       /
    """

    def __init__(
        self,
        embed_dim=256,
        encoder_depth=3,
        decoder_depth=3,
        num_heads=8,
        num_obj_tokens=100,
        conf_max=5,
    ):
        super().__init__()

        # RoPE for merged 64x64 grid; frequency matches SAM branch in default model
        self.rope = RotaryPositionEmbedding2D(frequency=100 * (36 / 64))
        self.position_getter = PositionGetter()

        self.num_heads = num_heads
        self.obj_token = nn.Parameter(torch.randn(1, num_obj_tokens, embed_dim))
        nn.init.normal_(self.obj_token, std=1e-6)

        self.pi3_projector = TokenProjector(c1=1024, embed_dim=embed_dim, p_drop=0.2)
        self.sam_projector = TokenProjector(c1=256, embed_dim=embed_dim, p_drop=0.2)
        self.merge_projector = TokenProjector(c1=embed_dim * 2, embed_dim=embed_dim, p_drop=0.2)

        self.trans_encoder = TransformerEncoder(
            depth=encoder_depth,
            embedding_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=2048,
            rope=self.rope,
        )

        self.trans_decoder = TwoWayTransformer(
            depth=decoder_depth,
            embedding_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=2048,
            rope=self.rope,
        )

        self.upsampling = UpSampling(
            transformer_dim=embed_dim,
            num_objs=num_obj_tokens,
            use_high_res_features=True,
        )

        self.motion_head = MLP(embed_dim, embed_dim, 1, 3)
        self.motion_bias = nn.Parameter(torch.tensor(1.0))
        self.motion_weight = nn.Parameter(torch.empty(1))
        torch.nn.init.normal_(self.motion_weight, mean=0.0, std=0.01)

        self.conf_head = MLP(embed_dim, embed_dim, 1, 3)
        self.conf_max = conf_max

        self.iou_head = MLP(embed_dim, embed_dim, 1, 3)

    def forward(
        self,
        token: torch.Tensor,              # B, h, w, 1024
        sam_btn: torch.Tensor,             # B, 256, 64, 64
        high_res_features=None,            # [B,32,256,256], [B,64,128,128]
    ):
        B = token.shape[0]

        # 1. Pi3: project at native res, then resize to 64x64
        pi3_feat = self.pi3_projector(token.float())               # B, h, w, 256
        pi3_feat = pi3_feat.permute(0, 3, 1, 2)                   # B, 256, h, w
        pi3_feat = F.interpolate(pi3_feat, size=(64, 64),
                                 mode='bilinear', align_corners=False)  # B, 256, 64, 64

        # 2. SAM: project
        sam_feat = sam_btn.float().permute(0, 2, 3, 1)             # B, 64, 64, 256
        sam_feat = self.sam_projector(sam_feat)                     # B, 64, 64, 256
        sam_feat = sam_feat.permute(0, 3, 1, 2)                    # B, 256, 64, 64

        # 3. Concat + merge
        merged = torch.cat([pi3_feat, sam_feat], dim=1)            # B, 512, 64, 64
        merged = merged.permute(0, 2, 3, 1)                        # B, 64, 64, 512
        merged = self.merge_projector(merged)                      # B, 64, 64, 256
        merged = merged.permute(0, 3, 1, 2)                        # B, 256, 64, 64

        dense_pe = self.position_getter(B, 64, 64, device=merged.device).permute(0, 2, 1).view(B, -1, 64, 64)

        # 4. Encoder: self-attention
        merged = self.trans_encoder(merged, dense_pe)              # B, 256, 64, 64

        # 5. Decoder: two-way cross-attention
        obj_token = self.obj_token.repeat(B, 1, 1)                 # B, N, 256
        obj_token_out, dense_out = self.trans_decoder(merged, dense_pe, obj_token)
        # obj_token_out: B, N, 256
        # dense_out:     B, 256, 64, 64

        # 6. Upsampling -> masks
        mask_out = self.upsampling(dense_out, obj_token_out, high_res_features=high_res_features)

        motion_logits = self.motion_head(obj_token_out)
        motion_factor = motion_logits.sigmoid()
        # Motion-regulated mask: per-object additive offset driven by motion_logits, scaled by
        # learned motion_weight and shifted by motion_bias (both nn.Parameters). Suppresses static
        # objects (motion_factor < motion_bias) and boosts dynamic ones at the logit level.
        mask = mask_out + (self.motion_weight * (motion_factor - self.motion_bias))[..., None]
        iou_pred = self.iou_head(obj_token_out)
        conf_pred = torch.sigmoid(self.conf_head(obj_token_out)) * (self.conf_max - 1) + 1

        return {
            "mask": mask,
            "motion_logits": motion_logits,
            "iou_pred": iou_pred,
            "conf_pred": conf_pred,
        }
