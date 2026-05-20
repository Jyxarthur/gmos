"""GMOS-S: foreground/background variant of the GMOS proposer.

Shares the Pi3 + SAM2 feature fusion and self-attention encoder with
`GMOSProposer`, but replaces the object-query decoder with a single
convolutional decode head that outputs one binary foreground mask per frame.
Used for streaming inference where multi-object tracking is not required.
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.transformer import TransformerEncoder
from .layers.rope import RotaryPositionEmbedding2D, PositionGetter
from .layers.basic import LayerNorm2d, MLP
from .layers.projector import TokenProjector

logger = logging.getLogger(__name__)


class GMOS_S(nn.Module):
    """GMOS-S: foreground/background variant.

    Shares the encoder pipeline with GMOSProposer (Pi3 + SAM2 features merged
    at 64×64, then transformer self-attention). Replaces the query-based
    decoder with a convolutional decode head that outputs a single
    B×1×256×256 foreground mask per frame.
    """

    def __init__(
        self,
        embed_dim=256,
        encoder_depth=3,
        num_heads=8,
        conf_max=5,
    ):
        super().__init__()

        self.rope = RotaryPositionEmbedding2D(frequency=100 * (36 / 64))
        self.position_getter = PositionGetter()

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

        # Fg decode head: ConvTranspose chain (64×64 → 256×256) + 1×1 conv
        self.fg_upscaling = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(embed_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
        )
        self.fg_out = nn.Conv2d(embed_dim // 8, 1, kernel_size=1)
        self.fg_out.weight.data = self.fg_out.weight.data.contiguous()

        # Auxiliary heads (operate on global-average-pooled encoder features)
        self.iou_head = MLP(embed_dim, embed_dim, 1, 3)
        self.conf_head = MLP(embed_dim, embed_dim, 1, 3)
        self.conf_max = conf_max

    def forward(
        self,
        token: torch.Tensor,              # B, h, w, 1024
        sam_btn: torch.Tensor,             # B, 256, 64, 64
        high_res_features=None,            # [B,32,256,256], [B,64,128,128]
    ):
        B = token.shape[0]

        # 1. Pi3: project then resize to 64×64
        pi3_feat = self.pi3_projector(token.float())               # B, h, w, 256
        pi3_feat = pi3_feat.permute(0, 3, 1, 2)                   # B, 256, h, w
        pi3_feat = F.interpolate(pi3_feat, size=(64, 64),
                                 mode='bilinear', align_corners=False)

        # 2. SAM: project
        sam_feat = sam_btn.float().permute(0, 2, 3, 1)             # B, 64, 64, 256
        sam_feat = self.sam_projector(sam_feat)                     # B, 64, 64, 256
        sam_feat = sam_feat.permute(0, 3, 1, 2)                    # B, 256, 64, 64

        # 3. Concat + merge
        merged = torch.cat([pi3_feat, sam_feat], dim=1)
        merged = merged.permute(0, 2, 3, 1)
        merged = self.merge_projector(merged)
        merged = merged.permute(0, 3, 1, 2)                       # B, 256, 64, 64

        # 4. Position encoding (RoPE)
        dense_pe = self.position_getter(B, 64, 64, device=merged.device).permute(0, 2, 1).view(B, -1, 64, 64)

        # 5. Encoder
        merged = self.trans_encoder(merged, dense_pe)              # B, 256, 64, 64

        # 6. Fg decode head with high-res skip connections
        dc1, ln1, act1, dc2, act2 = self.fg_upscaling
        feat_s0, feat_s1 = high_res_features
        x = act1(ln1(dc1(merged) + feat_s1))                       # B, 64, 128, 128
        x = act2(dc2(x) + feat_s0)                                 # B, 32, 256, 256
        fg_mask = self.fg_out(x)                                   # B, 1, 256, 256

        # Global feature for auxiliary heads
        global_feat = merged.flatten(2).mean(dim=2)                # B, 256
        iou_pred = self.iou_head(global_feat)                      # B, 1
        conf_pred = torch.sigmoid(self.conf_head(global_feat)) * (self.conf_max - 1) + 1

        return {
            "fg_mask": fg_mask,
            "iou_pred": iou_pred,
            "conf_pred": conf_pred,
        }
