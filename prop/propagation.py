"""Two-stage SAM2 mask propagation driven by per-frame proposer outputs.

  - `propagate`         — stage 1: dynamic prompt injection from one start
                          frame, in a single direction.
  - `reduce_propagate`  — stage 2: static prompts (top-k frames by predicted
                          IoU from stage 1), in a single direction.
  - `smooth_motions`    — temporal smoothing of per-frame motion scores.

Used by `inference_gmos.run_propagator` to convert per-frame mask proposals
into temporally consistent object tracks.
"""
import numpy as np
import torch
import tqdm

from . import prop_config as config
from .prop_utils import suppress_tqdm, compute_iou, compute_precision, batch_hungarian_match, get_torch_hard_max


def _compute_motion_labels(out_mask_logits, pred_motion_mask, pred_masks, pred_motions, n_pred):
    """
    Compute two motion signals for each SAM2 output object:
      - motion_global: precision of SAM2 mask vs union of predicted motion masks
      - motion_per_obj: motion score inherited from best-overlapping predicted object
    """
    motion_global = (
        compute_precision((out_mask_logits > 0)[:, 0], pred_motion_mask[None])
        .float().cpu().numpy() > config.MOTION_PRECISION_THRES
    ).astype(np.float32)

    motion_per_obj = []
    for obj_idx in range(out_mask_logits.shape[0]):
        if n_pred == 0:
            motion_per_obj.append(0)
            continue
        cross_precisions = compute_precision(
            (out_mask_logits > 0)[obj_idx:obj_idx + 1, 0], pred_masks
        ).float().cpu().numpy()
        best_precision = np.max(cross_precisions)
        best_idx = np.argmax(cross_precisions)
        if best_precision > config.MOTION_PRECISION_THRES:
            motion_per_obj.append(pred_motions[best_idx].item())
        else:
            motion_per_obj.append(0)

    return motion_global, np.stack(motion_per_obj, 0)


def propagate(predictor, all_pred_masks, all_motions, all_ious, all_pred_motion_masks,
              generator, video_results, ranges, max_obj_idx, num_prompts, prompt_set,
              inference_state, reverse=False, device=None):
    """
    Stage 1 propagation with dynamic prompt injection.

    During propagation, new SAM2 prompts are injected when:
    1. An existing object's Hungarian-matched mask has IoU > MATCH_IOU_THRES with SAM2
       output AND the object is moving AND has high predicted IoU (reinforcement).
    2. A predicted mask has low precision against all SAM2 outputs (new object discovery).

    When prompts are injected, the SAM2 generator is restarted from the current frame.
    """
    dev = device if device is not None else torch.device("cuda")
    for out_frame_idx in tqdm.tqdm(range(ranges[0], ranges[1], ranges[2]), total=abs(ranges[1] - ranges[0])):
        with suppress_tqdm():
            _, out_obj_ids, out_mask_logits, out_obj_logits = next(generator)

        update = False
        pred_masks = all_pred_masks[out_frame_idx]
        n_pred, h, w = pred_masks.shape[-3:]
        n_sam = out_mask_logits.shape[0]
        pred_motions = torch.tensor(all_motions[out_frame_idx, :n_pred]).to(dev)
        pred_ious = torch.tensor(all_ious[out_frame_idx, :n_pred]).to(dev)

        sam_masks = out_mask_logits[:, 0].sigmoid()
        pred_masks_t = torch.tensor(pred_masks).to(dev)

        # Zero-pad predictions if SAM2 has more objects
        if n_sam > n_pred:
            pad = n_sam - n_pred
            pred_masks_t = torch.cat([pred_masks_t, torch.zeros(pad, h, w, device=dev)], 0)
            pred_motions = torch.cat([pred_motions, torch.zeros(pad, device=dev)], 0)
            pred_ious = torch.cat([pred_ious, torch.zeros(pad, device=dev)], 0)

        # Hungarian match predictions to SAM2 outputs
        pred_masks_hm, _, matched_tensors = batch_hungarian_match(
            pred_masks_t.unsqueeze(0), sam_masks.unsqueeze(0),
            [pred_motions.unsqueeze(0), pred_ious.unsqueeze(0)],
        )
        pred_masks_hm = pred_masks_hm[0]
        pred_motions_hm = matched_tensors[0][0]
        pred_ious_hm = matched_tensors[1][0]

        min_objs = min(pred_masks_hm.shape[0], sam_masks.shape[0])
        matched_ious = compute_iou(pred_masks_hm[:min_objs], sam_masks[:min_objs])

        # Dynamic prompt injection
        for obj_idx in range(pred_masks_hm.shape[0]):
            is_moving = pred_motions_hm[obj_idx] > config.MOTION_THRES
            is_confident = pred_ious_hm[obj_idx] > config.IOU_THRES
            if not (is_moving and is_confident):
                continue

            # Case 1: Reinforce existing object (high match IoU)
            if obj_idx < matched_ious.shape[0] and matched_ious[obj_idx] > config.MATCH_IOU_THRES:
                # Skip if too close to an existing prompt for this object
                existing_frames = np.array(list(prompt_set.keys()))
                same_obj = existing_frames[np.where(existing_frames[:, 1] == obj_idx)[0], 0]
                if np.abs(out_frame_idx - same_obj).min() < config.MIN_PROMPT_FRAME_DISTANCE:
                    continue
                _, out_obj_ids, out_mask_logits, out_obj_logits = predictor.add_new_mask(
                    inference_state=inference_state, frame_idx=out_frame_idx,
                    obj_id=obj_idx, mask=pred_masks_hm[obj_idx], with_object_ious=True,
                )
                prompt_set[(out_frame_idx, obj_idx)] = pred_masks_hm[obj_idx]
                num_prompts += 1
                update = True
                continue

            # Case 2: Add new object (low precision = not covered by SAM2)
            max_precision = compute_precision(pred_masks_hm[obj_idx:obj_idx + 1], sam_masks).max()
            if max_precision < config.MIN_PRECISION_THRES and max_obj_idx < config.MAX_OBJECTS:
                _, out_obj_ids, out_mask_logits, out_obj_logits = predictor.add_new_mask(
                    inference_state=inference_state, frame_idx=out_frame_idx,
                    obj_id=max_obj_idx, mask=pred_masks_hm[obj_idx], with_object_ious=True,
                )
                prompt_set[(out_frame_idx, max_obj_idx)] = pred_masks_hm[obj_idx]
                max_obj_idx += 1
                num_prompts += 1
                update = True

        # Restart generator if prompts were added (and not at the boundary)
        at_boundary = (reverse and out_frame_idx - 1 == 0) or (not reverse and out_frame_idx + 1 == ranges[1])
        if update and not at_boundary:
            generator.close()
            next_frame = out_frame_idx - 1 if reverse else out_frame_idx + 1
            generator = predictor.propagate_in_video(
                inference_state, start_frame_idx=next_frame, reverse=reverse, with_object_ious=True,
            )

        # Compute motion labels and store results
        pred_motion_mask = torch.tensor(all_pred_motion_masks[out_frame_idx]).to(dev)
        out_mask_logits = get_torch_hard_max(out_mask_logits.transpose(0, 1)).transpose(0, 1)
        motion_global, motion_per_obj = _compute_motion_labels(
            out_mask_logits, pred_motion_mask, torch.tensor(pred_masks).to(dev), pred_motions, n_pred,
        )

        video_results["mov_segments"][out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() * motion_global[i]
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        video_results["full_segments"][out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        video_results["motions"][out_frame_idx] = {
            out_obj_id: motion_global[i][None]
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        video_results["motions_alter"][out_frame_idx] = {
            out_obj_id: motion_per_obj[i][None]
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        video_results["object_ious"][out_frame_idx] = {
            out_obj_id: out_obj_logits[i].float().cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    return video_results, generator, max_obj_idx, num_prompts, prompt_set, inference_state


def reduce_propagate(all_pred_masks, all_motions, all_pred_motion_masks,
                     generator, video_results, ranges, inference_state, device=None):
    """
    Stage 2 propagation — no prompt injection, just mask propagation and motion scoring.
    """
    dev = device if device is not None else torch.device("cuda")
    for out_frame_idx in tqdm.tqdm(range(ranges[0], ranges[1], ranges[2]), total=abs(ranges[1] - ranges[0])):
        with suppress_tqdm():
            _, out_obj_ids, out_mask_logits = next(generator)

        out_mask_logits = get_torch_hard_max(out_mask_logits.transpose(0, 1)).transpose(0, 1)
        pred_motion_mask = torch.tensor(all_pred_motion_masks[out_frame_idx]).to(dev)

        pred_masks = torch.tensor(all_pred_masks[out_frame_idx]).to(dev)
        n_pred = pred_masks.shape[0]
        pred_motions = torch.tensor(all_motions[out_frame_idx, :n_pred]).to(dev)

        motion_global, motion_per_obj = _compute_motion_labels(
            out_mask_logits, pred_motion_mask, pred_masks, pred_motions, n_pred,
        )

        video_results["mov_segments"][out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() * motion_global[i]
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        video_results["full_segments"][out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        video_results["motions"][out_frame_idx] = {
            out_obj_id: motion_global[i][None]
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        video_results["motions_alter"][out_frame_idx] = {
            out_obj_id: motion_per_obj[i][None]
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    return video_results, generator, inference_state


def smooth_motions(motions_alter, weights):
    """
    Temporal smoothing with a 5-frame sliding window.
    weights: list of 5 floats for offsets [-2, -1, 0, +1, +2].
    """
    shifted = [np.copy(motions_alter) for _ in range(5)]
    for offset_idx, arr in enumerate(shifted):
        offset = offset_idx - 2
        if offset < 0:
            arr[:offset] = motions_alter[-offset:]
        elif offset > 0:
            arr[offset:] = motions_alter[:-offset]
    stacked = np.stack(shifted, 0)
    w = np.array(weights)
    return (stacked * w[:, None, None]).sum(0) / (w[:, None, None].sum(0) + 1e-8)
