"""
Pi3 encoder hook for extracting internal features without modifying source code.
Uses register_forward_hook on decoder BlockRope layers.

Two modes:
  - 'attn': Extract cross-frame attention maps from layers [7,15,23,31]
             Output: (B, h, w, 512) matching VGGT format
             512 = 4 layers * 16 heads * (N-1) frames * 2 views
  - 'feat': Extract mid-frame token features from layers [7,15,23,31]
             Output: (B, h, w, 4096)
             4096 = 4 layers * 1024 dim
"""

import torch


class Pi3Hook:
    """Hook-based feature extractor for Pi3Encoder's decoder blocks.

    Usage:
        hook = Pi3Hook(pi3_encoder, mode='attn')  # or mode='feat'
        hook.set_meta(B=2, N=5, h=37, w=37)
        output = pi3_encoder(imgs)                 # normal forward pass
        hooked_feats = hook.get_features()          # (B, h, w, C)
        hook.remove()                               # cleanup
    """

    def __init__(self, pi3_encoder, mode='attn', hook_layers=(7, 15, 23, 31)):
        assert mode in ('attn', 'feat'), f"mode must be 'attn' or 'feat', got {mode}"
        for l in hook_layers:
            assert l % 2 == 1, f"hook_layers must be odd (cross-frame), got {l}"

        self.encoder = pi3_encoder
        self.mode = mode
        self.hook_layers = list(hook_layers)
        self.patch_start_idx = pi3_encoder.patch_start_idx  # 5 register tokens

        self._storage = {}   # layer_idx -> stored tensor
        self._hooks = []
        self._meta = {}      # B, N, h, w set before forward

        self._register_hooks()

    def _register_hooks(self):
        for layer_idx in self.hook_layers:
            blk = self.encoder.decoder[layer_idx]
            if self.mode == 'attn':
                hook = blk.register_forward_hook(self._make_attn_hook(layer_idx, blk))
            else:
                hook = blk.register_forward_hook(self._make_feat_hook(layer_idx))
            self._hooks.append(hook)

    # ------------------------------------------------------------------
    # Reconstruct xpos (RoPE positions) for odd layers
    # Matches pi3_encoder.py decode(): position_getter + shift + register padding
    # ------------------------------------------------------------------
    def _reconstruct_xpos(self, B, N, h, w, device):
        """Reconstruct the position tensor for odd (cross-frame) layers.

        Returns: (B, N * hw_full, 2) int64 tensor, where hw_full = patch_start_idx + h*w
        """
        # position_getter: cartesian_prod(range(h), range(w)) → (1, h*w, 2) int64
        y = torch.arange(h, device=device)
        x = torch.arange(w, device=device)
        pos = torch.cartesian_prod(y, x).view(1, h * w, 2).expand(B * N, -1, 2).clone()
        # pos: (B*N, h*w, 2)

        # Shift by 1 (register tokens get pos=0, patches get pos>=1)
        pos = pos + 1

        # Prepend zeros for register tokens
        ps = self.patch_start_idx
        pos_special = torch.zeros(B * N, ps, 2, device=device, dtype=pos.dtype)
        pos = torch.cat([pos_special, pos], dim=1)  # (B*N, hw_full, 2)

        # For odd layers: reshape to (B, N*hw_full, 2)
        hw_full = ps + h * w
        pos = pos.reshape(B, N * hw_full, 2)

        return pos

    # ------------------------------------------------------------------
    # Attention mode: recompute Q@K^T and process like VGGT
    # ------------------------------------------------------------------
    def _make_attn_hook(self, layer_idx, blk):
        def hook_fn(module, input, output):
            # input is (hidden,) — xpos is passed as kwarg and NOT in input tuple
            x = input[0]  # (B, N*hw_full, 1024)
            B = x.shape[0]
            N = self._meta['N']
            h = self._meta['h']
            w = self._meta['w']

            # Reconstruct xpos since it's not available in hook input
            xpos = self._reconstruct_xpos(B, N, h, w, x.device)

            with torch.no_grad():
                attn_map = self._compute_attn_map(blk, x, xpos, B, N, h, w)
            self._storage[layer_idx] = attn_map

        return hook_fn

    def _compute_attn_map(self, blk, x, xpos, B, N, h, w):
        """Recompute Q@K^T and process into VGGT-style attention maps.

        Returns: (B, num_heads, N-1, 2, h, w) on CPU
        """
        ps = self.patch_start_idx
        hw_full = ps + h * w

        # Apply norm1 (matches BlockRope.forward: self.attn(self.norm1(x), xpos=xpos))
        x_normed = blk.norm1(x)

        B_total, seq_len, C = x_normed.shape
        num_heads = blk.attn.num_heads
        head_dim = C // num_heads

        # Compute Q, K — exactly matching FlashAttentionRope.forward:
        #   qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)
        #   → (B, num_heads, 3, seq_len, head_dim)
        #   q, k, v = [qkv[:,:,i] for i in range(3)]  → each (B, num_heads, seq_len, head_dim)
        qkv = blk.attn.qkv(x_normed).reshape(B_total, seq_len, 3, num_heads, head_dim)
        qkv = qkv.transpose(1, 3)  # (B, num_heads, 3, seq_len, head_dim)
        q, k = qkv[:, :, 0], qkv[:, :, 1]  # each (B, num_heads, seq_len, head_dim)

        # QK norm
        v_dtype = qkv[:, :, 2].dtype
        q = blk.attn.q_norm(q).to(v_dtype)
        k = blk.attn.k_norm(k).to(v_dtype)

        # RoPE
        if blk.attn.rope is not None:
            q = blk.attn.rope(q, xpos)
            k = blk.attn.rope(k, xpos)

        # --- Only compute attention for mid-frame query tokens (saves memory) ---
        mid = N // 2
        P = hw_full  # tokens per frame (including registers)
        mid_start = mid * P
        mid_end = (mid + 1) * P

        q_mid = q[:, :, mid_start:mid_end, :]  # (B, num_heads, P, head_dim)
        scale = blk.attn.scale  # head_dim ** -0.5

        # Attention scores: (B, num_heads, P, N*P)
        attn = torch.matmul(q_mid, k.transpose(-2, -1)) * scale

        # # Softmax over key dimension (matches standard attention)
        # attn = attn.softmax(dim=-1)

        # Reshape to per-frame structure: (B, num_heads, P, N, P)
        attn = attn.reshape(B_total, num_heads, P, N, P)

        # Remove register tokens from query and key dims
        # query: skip first ps registers → (B, num_heads, h*w, N, P)
        attn = attn[:, :, ps:, :, :]
        # key: skip first ps registers per frame → (B, num_heads, h*w, N, h*w)
        attn = attn[:, :, :, :, ps:]

        # Permute to (B, num_heads, N, h*w_q, h*w_k)
        attn = attn.permute(0, 1, 3, 2, 4)

        # Remove mid-frame self-attention → keep N-1 other frames
        other_frames = torch.cat([attn[:, :, :mid, :, :],
                                  attn[:, :, mid + 1:, :, :]], dim=2)
        # (B, num_heads, N-1, h*w, h*w)

        # Two views (matching VGGT aggregator.py lines 324-325):
        #   view1: mean over query spatial dim → per key-location map
        #   view2: mean over key spatial dim → per query-location map
        view1 = other_frames.mean(dim=-2)  # (B, num_heads, N-1, h*w)
        view2 = other_frames.mean(dim=-1)  # (B, num_heads, N-1, h*w)

        # Stack views: (B, num_heads, N-1, 2, h*w) → reshape to (B, num_heads, N-1, 2, h, w)
        attn_out = torch.stack([view1, view2], dim=3)
        attn_out = attn_out.reshape(B_total, num_heads, N - 1, 2, h, w)

        return attn_out.detach().cpu()  # offload to CPU like VGGT

    # ------------------------------------------------------------------
    # Feature mode: capture block output for mid frame
    # ------------------------------------------------------------------
    def _make_feat_hook(self, layer_idx):
        def hook_fn(module, input, output):
            # At odd layers, output shape is (B, N*hw_full, C)
            x = output
            B = x.shape[0]
            N = self._meta['N']
            h = self._meta['h']
            w = self._meta['w']
            hw_full = x.shape[1] // N
            C = x.shape[-1]

            mid = N // 2
            # Reshape to per-frame: (B, N, hw_full, C)
            x_frames = x.reshape(B, N, hw_full, C)
            # Extract mid frame, remove registers: (B, h*w, C)
            mid_feat = x_frames[:, mid, self.patch_start_idx:, :]
            # Reshape to spatial: (B, h, w, C)
            mid_feat = mid_feat.reshape(B, h, w, C)
            self._storage[layer_idx] = mid_feat.detach()

        return hook_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_meta(self, B, N, h, w):
        """Set frame/spatial metadata before forward pass.
        Call this before pi3_encoder.forward() so hooks know the dimensions.

        Args:
            B: batch size
            N: number of frames (typically 5)
            h: patch height = H // 14
            w: patch width = W // 14
        """
        self._meta = {'B': B, 'N': N, 'h': h, 'w': w}

    def clear(self):
        """Clear stored features from previous forward pass."""
        self._storage.clear()

    def get_features(self):
        """Get hooked features after a forward pass.

        Returns:
            attn mode: (B, h, w, 512) — 4 layers * 16 heads * (N-1) * 2
            feat mode: (B, h, w, 4096) — 4 layers * 1024 dim
        """
        feats = [self._storage[l] for l in self.hook_layers]

        if self.mode == 'attn':
            # Each feat: (B, 16, N-1, 2, h, w) on CPU
            # Stack layers: (B, 4, 16, N-1, 2, h, w)
            stacked = torch.stack(feats, dim=1)
            B = stacked.shape[0]
            h, w = stacked.shape[-2], stacked.shape[-1]
            # Flatten to (B, 4*16*(N-1)*2, h, w) then permute to (B, h, w, C)
            out = stacked.reshape(B, -1, h, w).permute(0, 2, 3, 1)
            return out  # (B, h, w, 512) for N=5

        else:  # feat mode
            # Each feat: (B, h, w, 1024) on GPU
            out = torch.cat(feats, dim=-1)  # (B, h, w, 4096)
            return out

    def remove(self):
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._storage.clear()

    def __del__(self):
        self.remove()


def hook_pi3_encoder(pi3_encoder, mode='attn', hook_layers=(7, 15, 23, 31)):
    """Convenience function to hook a Pi3Encoder.

    Args:
        pi3_encoder: Pi3Encoder instance
        mode: 'attn' or 'feat'
        hook_layers: tuple of odd layer indices to hook

    Returns:
        Pi3Hook instance

    Example:
        hook = hook_pi3_encoder(encoder, mode='attn')
        hook.set_meta(B=2, N=5, h=37, w=37)
        output = encoder(imgs)
        attn_maps = hook.get_features()  # (B, h, w, 512)
        hook.remove()
    """
    return Pi3Hook(pi3_encoder, mode=mode, hook_layers=hook_layers)
