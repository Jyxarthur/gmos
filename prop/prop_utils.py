"""Helpers used by the propagator and the saving step.

  - IoU + precision between predicted and SAM2-propagated masks.
  - Per-frame Hungarian matching with optional reordering of auxiliary tensors.
  - Hard-max one-hot conversion and result-array tensor assembly.
  - Palette-indexed PNG writing and DAVIS-palette overlay video assembly.
"""
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import cv2
from PIL import Image
from contextlib import contextmanager
from scipy.optimize import linear_sum_assignment


# ── tqdm suppression ──────────────────────────────────────────────────────────

@contextmanager
def suppress_tqdm():
    """Context manager to mute new tqdm instances created within the block."""
    original_init = tqdm.tqdm.__init__

    def patched_init(self, *args, **kwargs):
        kwargs['disable'] = True
        original_init(self, *args, **kwargs)

    tqdm.tqdm.__init__ = patched_init
    try:
        yield
    finally:
        tqdm.tqdm.__init__ = original_init


# ── Mask metrics ──────────────────────────────────────────────────────────────

def compute_iou(pred, gt, threshold=0.5):
    """Compute IoU between corresponding pred and gt masks. Shape: [..., H, W] -> [...]."""
    pred = (pred > threshold).float().reshape(*pred.shape[:-2], -1)
    gt = (gt > threshold).float().reshape(*gt.shape[:-2], -1)

    intersection = (pred * gt).sum(-1)
    union = pred.sum(-1) + gt.sum(-1) - intersection
    iou = intersection / (union + 1e-6)

    both_empty = (pred.sum(-1) == 0) & (gt.sum(-1) == 0)
    iou[both_empty] = 1.0
    return iou


def compute_precision(pred, gt, threshold=0.5):
    """Compute precision (intersection / pred_area). Shape: [..., H, W] -> [...]."""
    pred = (pred > threshold).float().reshape(*pred.shape[:-2], -1)
    gt = (gt > threshold).float().reshape(*gt.shape[:-2], -1)

    intersection = (pred * gt).sum(-1)
    return intersection / (pred.sum(-1) + 1e-6)


# ── Hungarian matching ────────────────────────────────────────────────────────

def batch_hungarian_match(pred, gt, tensors_to_reorder=None):
    """
    IoU-based Hungarian match between pred and gt masks.

    Args:
        pred: [B, N, H, W]
        gt:   [B, M, H, W]   (zero-padded to N if M < N)
        tensors_to_reorder: list of [B, N, ...] tensors to reorder along dim=1

    Returns:
        matched_pred, matched_idx, [reordered_tensors]
    """
    B, N, H, W = pred.shape
    N_gt = gt.shape[1]
    if N_gt < N:
        gt = torch.cat([gt, torch.zeros(B, N - N_gt, H, W, device=gt.device)], 1)

    iou_all = compute_iou(
        pred.sigmoid().detach().unsqueeze(2),
        gt.detach().unsqueeze(1),
    )  # [B, N, N]

    matched_pred = torch.zeros_like(pred)
    matched_idx = []
    reordered_tensors = (
        [torch.zeros_like(t) for t in tensors_to_reorder]
        if tensors_to_reorder is not None else None
    )

    for b in range(B):
        cost = 1 - iou_all[b].detach().cpu().numpy()
        row_idx, col_idx = linear_sum_assignment(cost)

        reordered = torch.zeros_like(pred[b])
        reordered[col_idx] = pred[b, row_idx]
        matched_pred[b] = reordered
        matched_idx.append(col_idx)

        if reordered_tensors is not None:
            for i, tensor in enumerate(tensors_to_reorder):
                reordered = torch.zeros_like(tensor[b])
                reordered[col_idx] = tensor[b, row_idx]
                reordered_tensors[i][b] = reordered

    if reordered_tensors is not None:
        return matched_pred, matched_idx, reordered_tensors
    return matched_pred, matched_idx


# ── Tensor utilities ─────────────────────────────────────────────────────────

def get_torch_hard_max(mask, threshold=-1):
    """Convert [B, C, H, W] soft masks to mutually-exclusive one-hot via argmax."""
    B, C, H, W = mask.shape
    max_values, argmax_result = torch.max(mask, dim=1)
    one_hot = F.one_hot(argmax_result, num_classes=C).permute(0, 3, 1, 2)
    confident_pixels = (max_values > threshold).unsqueeze(1)
    return one_hot.to(mask.dtype) * confident_pixels.to(mask.dtype)


def get_torch_priority_hard_max(mask, priority, threshold=0.0):
    """Mutually-exclusive one-hot resolved by object priority instead of logit value.

    Unlike `get_torch_hard_max` (per-pixel argmax over the channel value), this
    first thresholds each object's logits independently to obtain boolean masks,
    then resolves overlapping pixels by `priority`: the object with the larger
    priority value is painted on top (wins the pixel). Channels with no priority
    entry default to priority 0.

    Args:
        mask:      [B, C, H, W] per-object soft logits.
        priority:  1-D sequence/tensor of length C; higher = painted on top.
                   Pixels are assigned to argmax(priority) among the objects
                   whose logit exceeds `threshold` at that pixel.
        threshold: logit threshold for an object to be "present" at a pixel.

    Returns:
        [B, C, H, W] one-hot (at most one channel set per pixel), same dtype as mask.
    """
    B, C, H, W = mask.shape
    present = mask > threshold                               # [B, C, H, W] bool
    prio = torch.as_tensor(priority, device=mask.device, dtype=torch.float32)
    # Score each present object by its priority; absent objects get -inf so they
    # never win. argmax over channel then picks the highest-priority present obj.
    scored = torch.where(present, prio.view(1, C, 1, 1).expand(B, C, H, W),
                         torch.full_like(mask, float("-inf"), dtype=torch.float32))
    any_present = present.any(dim=1)                         # [B, H, W]
    argmax_result = scored.argmax(dim=1)                     # [B, H, W]; ties -> lowest index
    one_hot = F.one_hot(argmax_result, num_classes=C).permute(0, 3, 1, 2)
    return one_hot.to(mask.dtype) * any_present.unsqueeze(1).to(mask.dtype)


def form_result_array(video_result_dict, zero_info, total_num_frames):
    """Fill a pre-allocated array from a sparse {frame_idx: {obj_idx: value}} dict."""
    for frame_idx in range(total_num_frames):
        if frame_idx not in video_result_dict:
            continue
        for obj_idx, per_object_info in video_result_dict[frame_idx].items():
            zero_info[frame_idx, obj_idx] = per_object_info
    return zero_info


# ── Visualization / IO ───────────────────────────────────────────────────────

DAVIS_PALETTE_BYTES = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"


def _palette_from_bytes(palette_bytes):
    """Return a flat RGB list padded to 256*3 entries for PIL."""
    flat = list(palette_bytes)
    flat = flat[: (len(flat) // 3) * 3]
    if len(flat) < 256 * 3:
        flat += [0] * (256 * 3 - len(flat))
    elif len(flat) > 256 * 3:
        flat = flat[:256 * 3]
    return flat


def save_indexed_png(label, out_path):
    """Save a 2D label map as a paletted PNG using the DAVIS palette."""
    if label.ndim != 2:
        raise ValueError("label must be 2D [H, W]")
    if label.dtype != np.uint8:
        if label.max() >= 256:
            raise ValueError("label contains values >= 256")
        label = label.astype(np.uint8)

    pal = _palette_from_bytes(DAVIS_PALETTE_BYTES)
    n_colors = len(pal) // 3
    if label.max() >= n_colors:
        raise ValueError(f"Label index {label.max()} exceeds palette size ({n_colors}).")

    img = Image.fromarray(label, mode="P")
    img.putpalette(pal)
    img.save(out_path, format="PNG", optimize=True)


# DAVIS palette as (256, 3) np.uint8 array — same colors as save_indexed_png writes,
# so MP4 overlays use the same colors as the per-frame indexed PNGs.
DAVIS_PALETTE = np.array(_palette_from_bytes(DAVIS_PALETTE_BYTES), dtype=np.uint8).reshape(256, 3)


def overlay_masks(rgb, ann, alpha=0.5):
    """Overlay indexed annotation masks on RGB video frames using the DAVIS palette.
    Returns (T,H,W,3) uint8. Same colors as save_indexed_png so PNGs and MP4s agree."""
    assert rgb.dtype == np.uint8 and rgb.ndim == 4 and rgb.shape[-1] == 3
    assert ann.ndim == 3 and ann.shape[:3] == rgb.shape[:3]

    T, H, W, _ = rgb.shape
    ann_clamped = np.clip(ann, 0, len(DAVIS_PALETTE) - 1)
    mask_rgb = DAVIS_PALETTE[ann_clamped]

    fg = (ann_clamped != 0)[..., None]
    out = rgb.astype(np.float32)
    out[fg[..., 0]] = (1.0 - alpha) * out[fg[..., 0]] + alpha * mask_rgb[fg[..., 0]].astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)

    for t in range(T):
        a = ann_clamped[t]
        b = np.zeros((H, W), dtype=bool)
        b[1:, :] |= (a[1:, :] != a[:-1, :])
        b[:-1, :] |= (a[:-1, :] != a[1:, :])
        b[:, 1:] |= (a[:, 1:] != a[:, :-1])
        b[:, :-1] |= (a[:, :-1] != a[:, 1:])
        b &= (a != 0)

        frame_bgr = np.ascontiguousarray(out[t][..., ::-1])
        ys, xs = np.where(b)
        for y, x in zip(ys, xs):
            cv2.circle(frame_bgr, (int(x), int(y)), 0, (255, 255, 255), 1)
        out[t] = frame_bgr[..., ::-1]

    return out
