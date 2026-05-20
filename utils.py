"""Inference helpers shared between inference_gmos.py and inference_gmos_s.py.

  - get_numpy_hard_max         — per-pixel argmax → one-hot.
  - LoadedImageDirVideo, parse_input_rgb, resolve_fps_stride
                               — --input_rgb dispatch (mp4 file vs. directory of
                                 frames) and fps/stride resolution per video.
  - encode_pi3_with_cache      — Pi3 forward with DINOv2 outputs cached per frame
                                 index, so the same frame is encoded at most once.
  - init_state_lite, patch_predictor_cache_additive
                               — propagator helpers around the SAM2 video predictor:
                                 build an inference_state with prefilled features
                                 and make the per-frame feature cache additive.
"""
import os

import numpy as np
import torch
import mediapy as media
from PIL import Image


# ── argmax helper ────────────────────────────────────────────────────────────

def get_numpy_hard_max(mask, threshold=0):
    """One-hot per pixel of the argmax channel, masked to pixels whose max value
    exceeds `threshold`.

    Args:
        mask: (C, H, W) float array of per-channel scores.
        threshold: pixels with max(mask, dim=0) <= threshold get all-zero rows.

    Returns:
        (C, H, W) array with at most one channel set to 1 per pixel.
    """
    C, _, _ = mask.shape
    argmax_result = np.argmax(mask, axis=0)
    confident_pixels = np.max(mask, axis=0) > threshold
    one_hot = np.eye(C, dtype=mask.dtype)[argmax_result] * confident_pixels[..., np.newaxis]
    return one_hot.transpose(2, 0, 1)


# ── input_rgb dispatch + image-dir loader ────────────────────────────────────

_IMG_EXTS = (".jpg", ".jpeg", ".png")


class _LoadedVideoMeta:
    """Minimal stand-in for mediapy's VideoMetadata."""
    def __init__(self, fps):
        self.fps = fps


class LoadedImageDirVideo:
    """Drop-in replacement for the object returned by `mediapy.read_video()`
    when the source is a directory of JPEG/PNG frames.

    Exposes the same surface used by the rest of the pipeline:
      - .shape : (T, H, W, 3)
      - .metadata.fps : float (provided by caller, never read from disk)
      - __len__, __getitem__ : uint8 (H,W,3) per index or (k,H,W,3) for list-index
    """
    def __init__(self, dir_path, fps):
        fns = sorted(os.listdir(dir_path))
        fns = [f for f in fns if f.lower().endswith(_IMG_EXTS)]
        if not fns:
            raise ValueError(f"No JPEG/PNG frames found in {dir_path}")
        exts = {os.path.splitext(f)[1].lower() for f in fns}
        if len(exts) > 1:
            raise ValueError(f"Mixed image extensions in {dir_path}: {exts}. "
                             "All frames in one folder must share one extension.")
        self._paths = [os.path.join(dir_path, f) for f in fns]
        first = np.asarray(Image.open(self._paths[0]).convert("RGB"))
        self._H, self._W = first.shape[:2]
        self.shape = (len(self._paths), self._H, self._W, 3)
        self.metadata = _LoadedVideoMeta(float(fps))
        self._cache = {0: first}

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, (list, np.ndarray, tuple)):
            return np.stack([self._read(int(i)) for i in idx], axis=0)
        if isinstance(idx, slice):
            r = range(*idx.indices(len(self)))
            return np.stack([self._read(i) for i in r], axis=0)
        return self._read(int(idx))

    def __array__(self, dtype=None):
        arr = np.stack([self._read(i) for i in range(len(self))], axis=0)
        return arr.astype(dtype) if dtype is not None else arr

    def _read(self, i):
        if i in self._cache:
            return self._cache[i]
        return np.asarray(Image.open(self._paths[i]).convert("RGB"))


def _is_image_dir(path):
    if not os.path.isdir(path):
        return False
    return any(f.lower().endswith(_IMG_EXTS) for f in os.listdir(path))


def parse_input_rgb(paths):
    """Dispatch the --input_rgb list into [(kind, path, seq_name), ...].

    Accepts:
      (i)   single mp4 path
      (ii)  list of mp4 paths
      (iii) a directory of JPEGs/PNGs
      (iv)  a list of directories of JPEGs/PNGs

    All entries must be the SAME kind (all mp4, or all image-dirs). Mixed input
    or anything else (missing path, non-image dir, non-mp4 file) is a hard error.

    Returns: (kind, [(path, seq_name), ...]) where kind in {"mp4", "imgdir"}.
    """
    if not paths:
        raise ValueError("--input_rgb is empty")
    kinds = []
    resolved = []
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"--input_rgb entry does not exist: {p}")
        if os.path.isfile(p) and p.lower().endswith(".mp4"):
            kinds.append("mp4")
            resolved.append((p, os.path.basename(p).replace(".mp4", "")))
        elif _is_image_dir(p):
            kinds.append("imgdir")
            seq = os.path.basename(p.rstrip("/"))
            resolved.append((p, seq))
        else:
            raise ValueError(
                f"--input_rgb entry not recognized (not an .mp4 file, not a JPEG/PNG dir): {p}")
    kset = set(kinds)
    if len(kset) > 1:
        raise ValueError(f"--input_rgb mixes kinds ({kset}); pass either all mp4s "
                         "or all image directories, not a mix.")
    return kinds[0], resolved


def resolve_fps_stride(video_path, args_fps, args_stride, kind="mp4"):
    """Resolve (stride, fps_for_log, source) for one input.

    Priority:
      1. If args_stride is set -> use it directly, ignore fps entirely.
      2. Elif args_fps is set  -> stride = max(1, int(args_fps / 8)).
      3. Else (mp4 only)        -> read fps from video metadata, stride = max(1, int(fps / 8)).
                  (imgdir)       -> hard error: image directories have no metadata fps.

    Formula: stride = int(0.5 / (4 / fps)) = int(fps / 8).
    Returns (stride, fps_used_or_None, source_str).
    """
    if args_stride is not None and args_fps is not None:
        print(f"  [resolve_fps_stride] WARN: both --fps and --stride set; "
              f"--stride takes priority, ignoring --fps={args_fps}")
    if args_stride is not None:
        return int(args_stride), None, "stride-arg"
    if args_fps is not None:
        return max(1, int(args_fps / 8)), float(args_fps), "fps-arg"
    if kind == "imgdir":
        raise ValueError(
            f"image-dir input '{video_path}' has no metadata fps; "
            "specify --fps or --stride explicitly.")
    fps = float(media.read_video(video_path).metadata.fps)
    return max(1, int(fps / 8)), fps, "video-meta"


# ── Pi3 DINOv2 encoder cache ─────────────────────────────────────────────────

def encode_pi3_with_cache(pi3_encoder, pi3_rgbs, video_indices, pi3_cache, offload_cpu=False):
    """Pi3 forward with the DINOv2 encoder output cached per frame index.

    Because adjacent 5-frame windows overlap, the same video frame is fed to
    Pi3 multiple times within a single video. The DINOv2 encoder is
    deterministic per frame, so its output is cached keyed by `frame_idx` and
    only the BlockRope decoder runs per window.

    Args:
        pi3_encoder: Pi3Encoder instance (eval mode).
        pi3_rgbs: (B, N, C, H, W) un-normalized.
        video_indices: (B, N) long tensor of frame indices for each window position.
        pi3_cache: Dict[int -> torch.Tensor of shape (hw, 1024)] (mutated). Tensors
            are GPU-resident unless offload_cpu=True, in which case they are pinned-CPU.
        offload_cpu: if True, store new entries on pinned CPU memory and transfer
            back to GPU when assembling the decoder input.

    Returns:
        Pi3 decoder output for the input batch.
    """
    B, N, C, H, W = pi3_rgbs.shape

    # Normalize
    pi3_rgbs_normed = (pi3_rgbs - pi3_encoder.image_mean) / pi3_encoder.image_std

    # Determine which (b, n) positions need a fresh DINOv2 forward.
    # Cache by integer frame index. Boundary-clipped duplicates share entries.
    flat_indices = video_indices.reshape(-1).tolist()    # B*N ints
    uncached_indices = []   # global frame indices not yet in cache
    uncached_positions = [] # positions in B*N flat tensor that need encoding
    seen_during_this_call = {}  # local map: frame_idx -> position-in-uncached_indices

    for pos, fidx in enumerate(flat_indices):
        if fidx in pi3_cache:
            continue
        if fidx in seen_during_this_call:
            continue
        seen_during_this_call[fidx] = len(uncached_indices)
        uncached_indices.append(fidx)
        uncached_positions.append(pos)

    # Run DINOv2 only on uncached frames
    if uncached_indices:
        flat_imgs = pi3_rgbs_normed.reshape(B * N, C, H, W)
        uncached_imgs = flat_imgs[uncached_positions]
        encoder_out = pi3_encoder.encoder(uncached_imgs, is_training=True)
        if isinstance(encoder_out, dict):
            encoder_out = encoder_out["x_norm_patchtokens"]    # (uncached_count, hw, 1024)
        for k, fidx in enumerate(uncached_indices):
            feat = encoder_out[k]                              # (hw, 1024) on GPU
            if offload_cpu:
                # Move to pinned CPU memory for fast async H2D on subsequent reads
                feat = feat.detach().to("cpu", non_blocking=False).pin_memory()
            pi3_cache[fidx] = feat

    # Reassemble (B*N, hw, 1024) from cache. If entries live on CPU, transfer to GPU.
    if offload_cpu:
        device = pi3_rgbs.device
        hidden = torch.stack(
            [pi3_cache[fidx].to(device, non_blocking=True) for fidx in flat_indices], dim=0
        )
    else:
        hidden = torch.stack([pi3_cache[fidx] for fidx in flat_indices], dim=0)

    return pi3_encoder.decode(hidden, N, H, W)


# ── Lightweight SAM2 init_state replacement ──────────────────────────────────

def init_state_lite(predictor, num_frames, video_height, video_width, device):
    """Build a SAM2VideoPredictor inference_state with an empty feature cache.

    The caller is expected to prefill `inference_state["cached_features"]`
    before any propagation step. `inference_state["images"]` is set to a small
    placeholder tensor that supports `.expand`; the cache prefill covers every
    frame index in [0, num_frames) so the placeholder is never read.
    """
    from collections import OrderedDict

    inference_state = {}
    inference_state["images"] = torch.zeros(1, 3, predictor.image_size, predictor.image_size, device=device)
    inference_state["num_frames"] = num_frames
    inference_state["offload_video_to_cpu"] = False
    inference_state["offload_state_to_cpu"] = False
    inference_state["video_height"] = video_height
    inference_state["video_width"] = video_width
    inference_state["device"] = device
    inference_state["storage_device"] = device
    inference_state["point_inputs_per_obj"] = {}
    inference_state["mask_inputs_per_obj"] = {}
    inference_state["cached_features"] = {}
    inference_state["constants"] = {}
    inference_state["obj_id_to_idx"] = OrderedDict()
    inference_state["obj_idx_to_id"] = OrderedDict()
    inference_state["obj_ids"] = []
    inference_state["output_dict_per_obj"] = {}
    inference_state["temp_output_dict_per_obj"] = {}
    inference_state["frames_tracked_per_obj"] = {}
    return inference_state




# ── SAM2 per-frame feature cache: additive override ──────────────────────────

def patch_predictor_cache_additive(predictor):
    """Make the predictor's per-frame feature cache additive.

    By default the predictor replaces `inference_state["cached_features"]` on
    every cache miss, which would discard any frames we prefilled. After this
    patch, writes go through `cached_features[frame_idx] = ...` so prefilled
    entries persist across propagation calls.

    Additionally, if cache entries' tensors are on CPU (offload_cpu mode),
    this method transfers them to GPU before the downstream consumer reads
    them.
    """
    import types

    def _get_image_feature_additive(self, inference_state, frame_idx, batch_size):
        image, backbone_out = inference_state["cached_features"].get(frame_idx, (None, None))
        if backbone_out is None:
            device = inference_state["device"]
            image = inference_state["images"][frame_idx].to(device).float().unsqueeze(0)
            backbone_out = self.forward_image(image)
            inference_state["cached_features"][frame_idx] = (image, backbone_out)

        # Move CPU-offloaded backbone tensors back to GPU on demand.
        device = inference_state["device"]
        if backbone_out["backbone_fpn"][0].device.type == "cpu":
            backbone_out_gpu = {
                "backbone_fpn": [t.to(device, non_blocking=True) for t in backbone_out["backbone_fpn"]],
                "vision_pos_enc": [t.to(device, non_blocking=True) for t in backbone_out["vision_pos_enc"]],
            }
        else:
            backbone_out_gpu = backbone_out

        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out_gpu["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out_gpu["vision_pos_enc"].copy(),
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(
                batch_size, -1, -1, -1
            )
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            pos = pos.expand(batch_size, -1, -1, -1)
            expanded_backbone_out["vision_pos_enc"][i] = pos
        features = self._prepare_backbone_features(expanded_backbone_out)
        features = (expanded_image,) + features
        return features

    predictor._get_image_feature = types.MethodType(_get_image_feature_additive, predictor)
    return predictor
