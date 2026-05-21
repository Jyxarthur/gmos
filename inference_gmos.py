"""Merged GMOS inference: proposer + SAM2 mask propagator in one per-video pipeline.

For each video, runs the GMOS proposer to get initial per-frame object masks,
then runs the two-stage SAM2 video predictor propagation to produce temporally-
consistent tracks. No intermediate disk I/O between stages."""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import mediapy as media

# Local sam2 at core/sam2
_local_sam2 = os.path.join(os.path.dirname(__file__), "core", "sam2")
if _local_sam2 not in sys.path:
    sys.path.insert(0, _local_sam2)
from sam2.build_sam import build_sam2_video_predictor   # noqa: E402

from data.loader.video_loader import video_dataset      # noqa: E402
from core.pi3_encoder import Pi3Encoder                 # noqa: E402
from core.gmos_proposer import GMOSProposer             # noqa: E402

from utils import (                                     # noqa: E402
    LoadedImageDirVideo,
    encode_pi3_with_cache,
    get_numpy_hard_max,
    init_state_lite,
    parse_input_rgb,
    patch_predictor_cache_additive,
    resolve_fps_stride,
)
from prop import prop_config                            # noqa: E402
from prop.prop_utils import save_indexed_png, overlay_masks, form_result_array  # noqa: E402
from prop.propagation import propagate, reduce_propagate, smooth_motions  # noqa: E402


# ── Proposer: run on a single video, return per-frame in-memory results ───────

def run_proposer(video_path, pi3_encoder, predictor, gmos, device, batch_size=8,
                 sam_feat_cache=None, stride=3, use_dino_cache=True, offload_cpu=False,
                 preloaded_video=None, seq_name=None):
    """Run the GMOS proposer on a single video.

    Returns a dict with:
        seq_annos:    [T, H, W] uint8 indexed mask (0=bg, 1..M=objects)
        pred_motions: [T, N, 1] float (N=top-k object slots)
        pred_ious:    [T, N, 1] float

    T equals len(dataset) (one entry per 5-frame window mid-frame); M is the
    number of objects with non-zero mass.

    The proposer's SAM2 image encoder is the same module as the video
    predictor's (`predictor.forward_image`), so each frame is encoded once.

    If `sam_feat_cache` is provided as a dict, the SAM2 image-encoder
    `backbone_out` for each batch is written into it, keyed by per-frame global
    `mid_index`. The bottleneck level `backbone_fpn[-1]` is cloned to avoid
    an in-place `+= no_mem_embed` mutation corrupting the cached entry. The
    cache is later seeded into the SAM2 video predictor's `cached_features`.
    """
    dataset = video_dataset(video_path=video_path, strides=[stride], num_objs=20, sam_version="v2",
                            video=preloaded_video, seq_name=seq_name)
    val_loader = DataLoader(dataset, num_workers=8, batch_size=batch_size,
                            shuffle=False, pin_memory=True, drop_last=False)

    # SAM2 spatial feature sizes for reshape (256², 128², 64²) at image_size=1024.
    bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]

    gmos.eval()
    all_mid_indices = []
    all_annos = []
    all_motions = []
    all_ious = []
    pi3_cache = {}   # frame_idx -> DINOv2 patchtokens (hw, 1024), bf16/fp32

    for info in tqdm(val_loader, total=len(val_loader), desc="proposer"):
        sam_rgb = info["sam_rgb"].to(device)
        mid_indices = info["mid_index"]
        video_indices = info["video_indices"].to(device)        # (B, N) frame indices
        if "ori_hw" in info:
            ori_H, ori_W = info["ori_hw"][0, 0].item(), info["ori_hw"][0, 1].item()
        else:
            ori_H, ori_W = info["annos"].shape[-2], info["annos"].shape[-1]
        pi3_rgbs = info["pi3_rgbs"].to(device)
        T = pi3_rgbs.shape[1]

        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                if use_dino_cache:
                    pi3_hidden = encode_pi3_with_cache(pi3_encoder, pi3_rgbs, video_indices, pi3_cache,
                                                       offload_cpu=offload_cpu)
                else:
                    pi3_hidden = pi3_encoder(pi3_rgbs)
                pi3_hidden_mid = pi3_hidden[:, T // 2, ..., -1024:].detach().clone().float()

        with torch.no_grad():
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                # SAM2 image encoder forward (shared with propagator; weights loaded once).
                backbone_out = predictor.forward_image(sam_rgb)

                # Stash SAM2 features for propagator reuse (BEFORE _get_vision_features
                # mutates backbone_fpn[-1] in place via += no_mem_embed)
                if sam_feat_cache is not None:
                    B = sam_rgb.shape[0]
                    last_fpn_idx = len(backbone_out["backbone_fpn"]) - 1
                    for b in range(B):
                        # Slice backbone_out per batch element. Each FPN level is (B,C,H,W);
                        # clone bottleneck (last level) to detach from the about-to-mutate
                        # storage. Other levels: keep view (won't be mutated).
                        # If offload_cpu, move to pinned CPU memory immediately.
                        fpn_slices = []
                        for i, fpn in enumerate(backbone_out["backbone_fpn"]):
                            t = fpn[b:b+1].clone() if i == last_fpn_idx else fpn[b:b+1]
                            if offload_cpu:
                                # Need .clone() on non-bottleneck levels too before .to(cpu)
                                # because views into the original GPU tensor would still pin it.
                                t = t.clone() if i != last_fpn_idx else t
                                t = t.to("cpu", non_blocking=False).pin_memory()
                            fpn_slices.append(t)
                        pos_slices = []
                        for pos in backbone_out["vision_pos_enc"]:
                            t = pos[b:b+1]
                            if offload_cpu:
                                t = t.clone().to("cpu", non_blocking=False).pin_memory()
                            pos_slices.append(t)
                        per_sample = {
                            "backbone_fpn": fpn_slices,
                            "vision_pos_enc": pos_slices,
                        }
                        mid_idx = int(mid_indices[b].item() if hasattr(mid_indices[b], "item") else mid_indices[b])
                        sam_feat_cache[mid_idx] = per_sample

                # Equivalent of SAM2Encoder._prepare_backbone_features + _get_vision_features:
                # flatten the last 3 feature levels, add no_mem_embed to bottleneck, reshape.
                _, vision_feats, _, _ = predictor._prepare_backbone_features(backbone_out)
                # `directly_add_no_mem_embed` is True for SAM2.1 video predictor.
                vision_feats[-1] = vision_feats[-1] + predictor.no_mem_embed
                bs = vision_feats[0].shape[1]
                feats = [
                    feat.permute(1, 2, 0).view(bs, -1, *feat_size)
                    for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
                ][::-1]
                sam_btn = feats[-1].detach().clone().float()
                sam_highres = [f.float().detach().clone() for f in feats[:-1]]

            with torch.inference_mode():
                predictions = gmos(pi3_hidden_mid, sam_btn, sam_highres)
                masks_lr = predictions["mask"]
                motion_logits = predictions["motion_logits"]
                iou_pred = predictions["iou_pred"]

                # Top-K filtering
                topk = min(20, masks_lr.shape[1])
                _, topk_indices = motion_logits[..., 0].topk(topk, dim=1)
                masks_lr = torch.gather(masks_lr, 1, topk_indices[:, :, None, None].expand(-1, -1, masks_lr.shape[2], masks_lr.shape[3]))
                motion_logits = torch.gather(motion_logits, 1, topk_indices[:, :, None].expand(-1, -1, motion_logits.shape[2]))
                iou_pred = torch.gather(iou_pred, 1, topk_indices[:, :, None].expand(-1, -1, iou_pred.shape[2]))

                masks = F.interpolate(masks_lr, size=(ori_H, ori_W), mode="bilinear", align_corners=False)

        all_mid_indices.extend(mid_indices.detach().cpu().tolist())
        all_annos.append(masks.detach().cpu().numpy())
        all_ious.append(iou_pred.detach().cpu().numpy())
        all_motions.append(motion_logits.sigmoid().detach().cpu().numpy())

    all_annos = np.concatenate(all_annos, axis=0)      # (T, N, H, W) float
    all_motions = np.concatenate(all_motions, axis=0)  # (T, N, 1) float
    all_ious = np.concatenate(all_ious, axis=0)        # (T, N, 1) float
    T_seq, n_obj = all_annos.shape[:2]

    # Build seq_annos (per-frame indexed mask).
    seq_annos_oh = np.array([get_numpy_hard_max(anno) for anno in all_annos])  # (T, N, H, W) bool/int

    moving_obj_indices = np.where(seq_annos_oh.mean(axis=(0, -1, -2)) > 0)[0].tolist()
    obj_indices = np.zeros((T_seq, n_obj), dtype=np.int32)
    if moving_obj_indices:
        obj_indices[:, moving_obj_indices] = np.arange(len(moving_obj_indices))[None] + 1

    seq_annos = (
        np.clip(obj_indices[:, :, None, None], 0, None) * seq_annos_oh
    ).max(1).astype(np.int32)  # (T, H, W) int32, max object index per pixel

    pred_motions = np.zeros_like(all_motions)
    pred_ious = np.zeros_like(all_ious)
    for fi in range(T_seq):
        nonzero = np.where(obj_indices[fi] > 0)[0]
        pred_motions[fi, obj_indices[fi, nonzero] - 1] = all_motions[fi, nonzero]
        pred_ious[fi, obj_indices[fi, nonzero] - 1] = all_ious[fi, nonzero]

    return {
        "seq_annos": seq_annos.astype(np.uint8),  # uint8 to match disk roundtrip
        "pred_motions": pred_motions,
        "pred_ious": pred_ious,
        # Share the dataset's already-decoded frames so process_video doesn't need
        # to call media.read_video a second time for save_results.
        "frames": dataset.video,
    }


# ── In-memory tensor handoff into the propagator ─────────────────────────────

def build_propagator_inputs(seq_annos, pred_motions, pred_ious):
    """Build the (all_pred_masks, all_pred_motion_masks, all_motions, all_ious) tuple
    consumed by the propagator.

    Args:
        seq_annos:    [T, H, W] uint8 indexed mask (0=bg, 1..M=objects).
        pred_motions: [T, N, 1] float (proposer's per-object motion logits sigmoid).
        pred_ious:    [T, N, 1] float (proposer's per-object iou predictions).
    """
    T = seq_annos.shape[0]
    all_motions = pred_motions[..., 0]  # [T, N]
    all_ious = pred_ious[..., 0]        # [T, N]

    all_pred_masks = [
        seq_annos[t, None] == (np.arange(int(seq_annos[t].max())) + 1)[:, None, None]
        for t in range(T)
    ]  # list of (N_t, H, W) bool

    all_pred_motion_masks = np.stack([
        ((all_motions[t, :m.shape[0], None, None] > prop_config.MOTION_THRES) * m).sum(0)
        for t, m in enumerate(all_pred_masks)
    ], 0)  # [T, H, W]

    return {
        "all_ious": all_ious,
        "all_motions": all_motions,
        "all_pred_masks": all_pred_masks,
        "all_pred_motion_masks": all_pred_motion_masks,
        "total_num_frames": T,
    }


# ── Propagator: two-stage video propagation ──────────────────────────────────

def _init_video_results(with_object_ious=False):
    results = {
        "mov_segments": {},
        "full_segments": {},
        "motions": {},
        "motions_alter": {},
    }
    if with_object_ious:
        results["object_ious"] = {}
    return results


def select_start_frame(all_ious, all_motions):
    """Pick start frame for stage 1 propagation (forward strategy).
    Tries the first 3 frames if any of them has a high-quality moving object;
    otherwise falls back to the earliest valid frame."""
    all_frame_ious = all_ious * (all_ious > prop_config.IOU_THRES) * (all_motions > prop_config.MOTION_THRES)
    if all_frame_ious.sum() == 0:
        return None, all_frame_ious

    first3 = all_ious[0:3] * (all_ious[0:3] > prop_config.IOU_THRES) * (all_motions[0:3] > prop_config.MOTION_THRES)
    if first3.sum() == 0:
        start_frame = sorted(np.where(all_frame_ious.sum(-1) > 0)[0])[0]
    else:
        start_frame = np.argmax(first3.sum(-1))
    return start_frame, all_frame_ious


def run_propagator(predictor, video_path, frames_shape, prop_inputs, device,
                   sam_feat_cache=None, offload_cpu=False):
    """Two-stage SAM2 propagation. Returns (video_results_on, video_results_off).

    Propagation strategy:
      - stage 1 start frame: forward (prefer frames 0-2; fall back to the
        earliest frame with any valid mask).
      - stage 2 prompts: SAM2 prompts from the top-10% frames by object_ious
        from stage 1.
      - stage 2 start frame: the frame with the highest summed object_ious.

    If `sam_feat_cache` is supplied (a dict {frame_idx: backbone_out}), it is
    loaded into the predictor's `inference_state["cached_features"]` so SAM2
    never re-encodes any frame. Requires `predictor._get_image_feature` to
    have been wrapped by `patch_predictor_cache_additive(predictor)`.
    """
    H, W = frames_shape[1:3]

    all_ious = prop_inputs["all_ious"]
    all_motions = prop_inputs["all_motions"]
    all_pred_masks = prop_inputs["all_pred_masks"]
    all_pred_motion_masks = prop_inputs["all_pred_motion_masks"]
    total_num_frames = prop_inputs["total_num_frames"]

    # Lightweight init when sam_feat_cache covers every frame; fall back to
    # the standard init_state otherwise.
    if sam_feat_cache is not None and len(sam_feat_cache) == total_num_frames:
        inference_state = init_state_lite(
            predictor, num_frames=total_num_frames,
            video_height=H, video_width=W, device=device,
        )
    else:
        inference_state = predictor.init_state(video_path=video_path)

    # Prefill the SAM2 per-frame feature cache from the proposer's encodes.
    if sam_feat_cache is not None:
        # _get_image_feature only consumes `image` via `image.expand(batch_size, ...)`,
        # so all frames can share the same single-element placeholder tensor.
        image_placeholder = inference_state["images"][:1]
        for frame_idx, backbone_out in sam_feat_cache.items():
            inference_state["cached_features"][frame_idx] = (image_placeholder, backbone_out)

    # ── Stage 1: pick seed-prompt frame, seed initial prompts ─────────────
    on_start_frame, all_frame_ious = select_start_frame(all_ious, all_motions)
    if on_start_frame is None:
        print("No moving objects")
        return None, None

    max_obj_idx, num_prompts = 0, 0
    prompt_set = {}
    pred_masks_start = all_pred_masks[on_start_frame]
    for obj_idx in range(pred_masks_start.shape[0]):
        if all_motions[on_start_frame, obj_idx] > prop_config.MOTION_THRES and \
           all_ious[on_start_frame, obj_idx] > prop_config.IOU_THRES:
            predictor.add_new_mask(
                inference_state=inference_state, frame_idx=on_start_frame,
                obj_id=max_obj_idx, mask=pred_masks_start[obj_idx], with_object_ious=True,
            )
            prompt_set[(on_start_frame, max_obj_idx)] = pred_masks_start[obj_idx]
            max_obj_idx += 1
            num_prompts += 1

    # Seed prompts at on_start_frame; forward propagation always starts at
    # frame 0 so earlier frames are also covered.
    actual_start = 0
    print(f"prop 1 forward frame: {actual_start}")

    # ── Stage 1: propagate with dynamic prompt injection ──────────────────
    video_results_on = _init_video_results(with_object_ious=True)
    generator = predictor.propagate_in_video(
        inference_state, start_frame_idx=actual_start, reverse=False, with_object_ious=True,
    )
    video_results_on, generator, max_obj_idx, num_prompts, prompt_set, inference_state = propagate(
        predictor, all_pred_masks, all_motions, all_ious, all_pred_motion_masks,
        generator, video_results_on, (actual_start, total_num_frames, 1),
        max_obj_idx, num_prompts, prompt_set, inference_state, reverse=False, device=device,
    )
    if actual_start > 0:
        generator.close()
        generator = predictor.propagate_in_video(
            inference_state, start_frame_idx=actual_start, reverse=True, with_object_ious=True,
        )
        video_results_on, generator, max_obj_idx, num_prompts, prompt_set, inference_state = propagate(
            predictor, all_pred_masks, all_motions, all_ious, all_pred_motion_masks,
            generator, video_results_on, (actual_start, -1, -1),
            max_obj_idx, num_prompts, prompt_set, inference_state, reverse=True, device=device,
        )

    motions = form_result_array(video_results_on["motions"], np.zeros((total_num_frames, 10, 1)), total_num_frames)[:, :, 0]
    motions_alter = form_result_array(video_results_on["motions_alter"], np.zeros((total_num_frames, 10, 1)), total_num_frames)[:, :, 0]
    obj_ious = form_result_array(video_results_on["object_ious"], np.zeros((total_num_frames, 10, 1)), total_num_frames)[:, :, 0]
    full_segments = form_result_array(video_results_on["full_segments"], np.zeros((total_num_frames, 10, 1, H, W)), total_num_frames)[:, :, 0]

    motions_alter = smooth_motions(motions_alter, prop_config.STAGE1_SMOOTHING_WEIGHTS)
    video_results_on["mov_segments"] = {
        frame_idx: {
            obj_idx: obj_mask * (motions[frame_idx, obj_idx] > prop_config.MOTION_IOU_THRES)
            for obj_idx, obj_mask in frame_dict.items()
        }
        for frame_idx, frame_dict in video_results_on["full_segments"].items()
    }

    generator.close()
    predictor.reset_state(inference_state)

    # ── Stage 2: prompt selection from stage 1 top-IoU frames ────────────
    # Top-STAGE2_PROMPT_FRACTION of frames per object by stage 1 object_ious.
    prompt_set = {}
    for obj_idx in range(obj_ious.shape[1]):
        num_prompt = int(np.ceil((obj_ious[:, obj_idx] > 0).sum() * prop_config.STAGE2_PROMPT_FRACTION))
        if num_prompt == 0:
            continue
        selected = np.argpartition(obj_ious[:, obj_idx], -num_prompt)[-num_prompt:]
        for frame_idx in selected:
            if obj_ious[frame_idx, obj_idx] < prop_config.STAGE2_PROMPT_IOU_THRES:
                continue
            prompt_set[(frame_idx, obj_idx)] = full_segments[frame_idx, obj_idx]

    # Per-object prompt count: prop 2 prioritises objects with more stage-2
    # prompts when resolving overlapping masks (see reduce_propagate).
    prompt_counts = {}
    for (_, obj) in prompt_set:
        prompt_counts[obj] = prompt_counts.get(obj, 0) + 1

    for (t, obj), mask in prompt_set.items():
        predictor.add_new_mask(
            inference_state=inference_state, frame_idx=t, obj_id=obj, mask=mask,
        )

    # Select strategy: frame with highest summed object_ious across all objects.
    off_start_frame = obj_ious.sum(-1).argmax()
    print(f"prop 2 select frame: {off_start_frame}")

    # ── Stage 2: propagate without prompt injection ───────────────────────
    video_results_off = _init_video_results(with_object_ious=False)
    generator = predictor.propagate_in_video(
        inference_state, start_frame_idx=off_start_frame, reverse=False, with_object_ious=False,
    )
    video_results_off, generator, inference_state = reduce_propagate(
        all_pred_masks, all_motions, all_pred_motion_masks,
        generator, video_results_off, (off_start_frame, total_num_frames, 1), inference_state, device=device,
        prompt_counts=prompt_counts,
    )
    if off_start_frame > 0:
        generator.close()
        generator = predictor.propagate_in_video(
            inference_state, start_frame_idx=off_start_frame, reverse=True, with_object_ious=False,
        )
        video_results_off, generator, inference_state = reduce_propagate(
            all_pred_masks, all_motions, all_pred_motion_masks,
            generator, video_results_off, (off_start_frame, -1, -1), inference_state, device=device,
            prompt_counts=prompt_counts,
        )

    motions = form_result_array(video_results_off["motions"], np.zeros((total_num_frames, 10, 1)), total_num_frames)[:, :, 0]
    motions_alter = form_result_array(video_results_off["motions_alter"], np.zeros((total_num_frames, 10, 1)), total_num_frames)[:, :, 0]
    motions_alter = smooth_motions(motions_alter, prop_config.STAGE2_SMOOTHING_WEIGHTS)
    video_results_off["mov_segments"] = {
        frame_idx: {
            obj_idx: obj_mask * (motions[frame_idx, obj_idx] > prop_config.MOTION_IOU_THRES)
            for obj_idx, obj_mask in frame_dict.items()
        }
        for frame_idx, frame_dict in video_results_off["full_segments"].items()
    }

    # Filter full_segments: discard objects moving in <MIN_MOV_OBJECT_FRAME_PROP of frames
    static_objs = set(np.where(motions.mean(0) < prop_config.MIN_MOV_OBJECT_FRAME_PROP)[0])
    if static_objs:
        video_results_off["full_segments"] = {
            frame_idx: {
                obj_idx: obj_mask for obj_idx, obj_mask in frame_dict.items()
                if obj_idx not in static_objs
            }
            for frame_idx, frame_dict in video_results_off["full_segments"].items()
        }

    return video_results_on, video_results_off


# ── Saving final propagator outputs ───────────────────────────────────────────

def _build_indexed_masks(video_segments, T, H, W):
    """Build (T, H, W) uint8 indexed mask array from {frame_idx: {obj_id: mask}} dict."""
    all_masks = []
    for out_frame_idx in range(T):
        if out_frame_idx not in video_segments:
            all_masks.append(np.zeros((H, W), dtype=np.uint8))
        else:
            all_objs = np.zeros((H, W))
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                all_objs += out_mask[0] * (out_obj_id + 1)
            all_masks.append(all_objs.astype(np.uint8))
    return np.stack(all_masks, 0)


def _save_one_kind(all_masks, frames, fps, save_base_dir, kind, seq_name, save_mode):
    """Save one kind ('mos-i' or 'mos') in the requested modes.

    PNG output: <save_base_dir>/<kind>_frame/<seq_name>/00000.png ...
    MP4 output: <save_base_dir>/<kind>_mp4/<seq_name>.mp4
                (original video resolution, original fps, 0.5-alpha DAVIS-palette overlay)
    """
    if "mask" in save_mode:
        save_image_dir = f"{save_base_dir}/{kind}_frame/{seq_name}/"
        os.makedirs(save_image_dir, exist_ok=True)
        for idx in range(all_masks.shape[0]):
            save_indexed_png(all_masks[idx], os.path.join(save_image_dir, str(idx).zfill(5) + ".png"))

    if "mp4" in save_mode:
        save_video_dir = f"{save_base_dir}/{kind}_mp4/"
        os.makedirs(save_video_dir, exist_ok=True)
        all_rgb_masks = overlay_masks(np.asarray(frames), all_masks, alpha=0.5)
        media.write_video(
            os.path.join(save_video_dir, f"{seq_name}.mp4"),
            all_rgb_masks,
            fps=fps,
        )


def save_results(frames, video_results_on, video_results_off, save_base_dir, seq_name,
                 save_mode="mask", fps=30.0):
    """Save propagator outputs as PNG sequences (and optionally MP4s).

      - mos-i: stage 1 moving objects (after dynamic re-prompting).
      - mos:   stage 2 full multi-object segmentation.
    """
    T, H, W = frames.shape[:3]
    if video_results_on == {} and video_results_off == {}:
        video_results_on["mov_segments"] = {}
        video_results_off["full_segments"] = {}

    save_list = [
        ("mos-i", video_results_on["mov_segments"]),
        ("mos",   video_results_off["full_segments"]),
    ]
    for kind, video_segments in save_list:
        all_masks = _build_indexed_masks(video_segments, T, H, W)
        _save_one_kind(all_masks, frames, fps, save_base_dir, kind, seq_name, save_mode)


# ── Per-video pipeline ────────────────────────────────────────────────────────

def process_video(video_path, pi3_encoder, gmos, predictor, save_base_dir,
                  batch_size, device, save_mode="mask",
                  stride=3, use_dino_cache=True, offload_cpu=False,
                  preloaded_video=None, seq_name=None, output_fps=None):
    if seq_name is None:
        seq_name = os.path.basename(video_path).replace(".mp4", "")
    print(f"\n=== {seq_name} ===")

    # Stage A: proposer → seq_annos + motions + ious. The SAM2 image-encoder
    # `backbone_out` for each mid-frame is captured into `sam_feat_cache` for
    # reuse in the propagator.
    sam_feat_cache = {}
    proposer_out = run_proposer(
        video_path, pi3_encoder, predictor, gmos, device,
        batch_size=batch_size, sam_feat_cache=sam_feat_cache,
        stride=stride, use_dino_cache=use_dino_cache,
        offload_cpu=offload_cpu,
        preloaded_video=preloaded_video, seq_name=seq_name,
    )

    # Stage B: build propagator inputs in-memory.
    prop_inputs = build_propagator_inputs(
        proposer_out["seq_annos"], proposer_out["pred_motions"], proposer_out["pred_ious"]
    )

    # Stage C: reuse the proposer's already-decoded frames for the save step.
    frames = proposer_out["frames"]
    if output_fps is not None:
        fps = float(output_fps)
    elif hasattr(frames, "metadata"):
        fps = float(frames.metadata.fps)
    else:
        fps = 30.0

    # Stage D: run the propagator under bf16 autocast + tf32, scoped to this
    # block only so the proposer's matmul precision is unaffected.
    prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            video_results_on, video_results_off = run_propagator(
                predictor, video_path, frames.shape, prop_inputs, device,
                sam_feat_cache=sam_feat_cache,
                offload_cpu=offload_cpu,
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
        torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32
    if video_results_on is None:
        video_results_on, video_results_off = {}, {}

    # Stage E: save
    save_results(frames, video_results_on, video_results_off, save_base_dir, seq_name,
                 save_mode=save_mode, fps=fps)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pi3_ckpt", type=str, required=True,
                        help="Path to the Pi3 checkpoint (.safetensors or .pth).")
    parser.add_argument("--sam2_ckpt", type=str, required=True,
                        help="Path to the SAM2 Hiera-L checkpoint (.pt).")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gmos_ckpt", type=str, required=True,
                        help="Path to the trained GMOS proposer checkpoint (.pth).")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Output directory. Subdirs created inside: "
                             "{mos-i_frame, mos-i_mp4, mos_frame, mos_mp4}/<seq>/...")
    parser.add_argument("--input_rgb", type=str, nargs='+', required=True,
                        help="One or more inputs. Accepts: (i) a single .mp4 file, "
                             "(ii) multiple .mp4 files, (iii) a directory of JPEG/PNG frames, "
                             "(iv) multiple directories of JPEG/PNG frames. All entries must be "
                             "the same kind (all .mp4 or all image dirs). For image-dir inputs, "
                             "--fps or --stride must be specified explicitly.")
    parser.add_argument("--save_mode", type=str, default="mask",
                        choices=["mask", "mp4", "mask+mp4"],
                        help="What to write: 'mask' = per-frame indexed PNGs (DAVIS palette, "
                             "video resolution); 'mp4' = single MP4 per video with 0.5-alpha "
                             "DAVIS-palette overlay at original video resolution + FPS; 'mask+mp4' = both.")
    parser.add_argument("--stride", type=int, default=None,
                        help="Frame stride between window elements in the proposer's 5-frame Pi3 windows. "
                             "If unset, derived from --fps (or the video's metadata fps) via "
                             "stride = max(1, int(fps/8)). If both --stride and --fps are set, "
                             "--stride wins.")
    parser.add_argument("--fps", type=float, default=None,
                        help="Override the per-video fps used to compute --stride. Ignored if "
                             "--stride is also set. If neither is set, fps is read from the video "
                             "metadata per file. Formula: stride = max(1, int(fps/8)).")
    parser.add_argument("--offload_cpu", action="store_true",
                        help="Offload the DINOv2 and SAM2 feature caches to pinned "
                             "CPU memory; use for long videos when GPU memory is tight.")
    args = parser.parse_args()

    device = torch.device("cuda")

    # Propagation thresholds (MOTION_THRES, IOU_THRES, etc.) live as module-level
    # constants in prop/prop_config.py. Edit there if you need to tune them.

    # tf32 + bf16 autocast are scoped to run_propagator() only (see
    # process_video Stage D); enabling them globally would change the
    # proposer's matmul precision.

    # ── Load Pi3 encoder ──────────────────────────────────────────────────
    pi3_encoder = Pi3Encoder().to(device).eval()
    if args.pi3_ckpt.endswith(".safetensors"):
        from safetensors.torch import load_file
        weight = load_file(args.pi3_ckpt)
    else:
        weight = torch.load(args.pi3_ckpt, map_location=device, weights_only=False)
    pi3_encoder.load_state_dict(weight, strict=False)
    pi3_encoder.eval()
    for p in pi3_encoder.parameters():
        p.requires_grad = False

    # ── Load GMOSProposer ─────────────────────────────────────────────────
    gmos = GMOSProposer().to(device)
    print("Loading GMOS ckpt")
    state_dict = torch.load(args.gmos_ckpt)["model_state_dict"]
    if any(k.startswith("module.") for k in state_dict.keys()):
        from collections import OrderedDict
        state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    gmos.load_state_dict(state_dict, strict=False)

    # ── Build SAM2 video predictor (for propagator) ───────────────────────
    SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(SAM2_MODEL_CFG, args.sam2_ckpt, device=device)

    # Make the predictor's per-frame feature cache additive so the prefill
    # from the proposer survives propagation.
    patch_predictor_cache_additive(predictor)

    # ── Output dir ────────────────────────────────────────────────────────
    save_base_dir = args.save_dir
    print(f"save_base_dir = {save_base_dir}  (save_mode={args.save_mode})")
    os.makedirs(save_base_dir, exist_ok=True)

    # ── Resolve --input_rgb ───────────────────────────────────────────────
    kind, entries = parse_input_rgb(args.input_rgb)
    print(f"--input_rgb: kind={kind}, n={len(entries)}")

    for path, seq_name in entries:
        stride, fps_used, source = resolve_fps_stride(path, args.fps, args.stride, kind=kind)
        print(f"[{seq_name}] fps_source={source} fps={fps_used} stride={stride}")

        preloaded = None
        output_fps = None
        if kind == "imgdir":
            output_fps = args.fps if args.fps is not None else 24.0
            preloaded = LoadedImageDirVideo(path, fps=output_fps)
        process_video(
            path, pi3_encoder, gmos, predictor, save_base_dir,
            batch_size=args.batch_size, device=device, save_mode=args.save_mode,
            stride=stride, offload_cpu=args.offload_cpu,
            preloaded_video=preloaded, seq_name=seq_name, output_fps=output_fps,
        )

    print("All done.")
