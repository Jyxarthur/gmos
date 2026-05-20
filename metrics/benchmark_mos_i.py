"""Motion-aware temporal IoU evaluation.

All inputs are CLI args:
  --dataset {davis17-im, ytvos19-im}
                     If --time_anno_csv is a merged multi-dataset CSV with
                     `dataset`/`split` columns, this picks the right slice
                     (davis17-im → davis, ytvos19-im → ytvos; split is always test).
  --anno_dir         palette-indexed GT mask root: <anno_dir>/<seq>/00000.png ...
  --time_anno_csv    per-seq motion frame ranges (2-column `seq_name,time_anno`,
                     or a merged multi-dataset CSV — see --dataset)
  --res_dir          per-seq prediction root, same layout as --anno_dir

Rows with empty `time_anno` (`{}`) are dropped from the CSV before iterating.
"""
import argparse
import ast
import os

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


# Map --dataset CLI value → (dataset_column_value, split_column_value) used to
# filter a merged multi-dataset time_anno CSV. split is hardcoded to "test".
DATASET_TO_CSV_SLICE = {
    "davis17-im": ("davis", "test"),
    "ytvos19-im": ("ytvos", "test"),
}


def obtain_dynamic_labels(data_dict, n_frames, n_objs=10):
    keys = sorted(data_dict.keys(), key=int)
    labels = torch.zeros(n_frames, n_objs)
    for key in keys:
        for sub_range in data_dict[key]:
            labels[sub_range[0]:sub_range[1]+1, int(key)-1] = 1
    return labels


def compute_iou(pred, gt, threshold=0.5):
    """
    Compute IoU between each corresponding pred and gt mask.

    Args:
        pred: [..., H, W]
        gt:   [..., H, W]

    Returns:
        iou: [...]
    """
    pred = (pred > threshold).float().reshape(*pred.shape[:-2], -1)
    gt = (gt > threshold).float().reshape(*gt.shape[:-2], -1)

    intersection = (pred * gt).sum(-1)
    union = pred.sum(-1) + gt.sum(-1) - intersection
    iou = intersection / (union + 1e-6)

    pred_area = pred.sum(-1)
    gt_area = gt.sum(-1)
    both_empty = (pred_area == 0) & (gt_area == 0)
    iou[both_empty] = 1.0

    return iou


def batch_hungarian_match(pred, gt):
    """
    Vectorized IoU + per-sample Hungarian match.

    Args:
        pred: [B, N, H, W] or [B, T, N, H, W]
        gt:   [B, N, H, W] or [B, T, N, H, W]

    Returns:
        matched_pred: same shape as pred
        matched_idx: list of [N] arrays
    """
    B = pred.shape[0]

    if pred.ndim == 5:
        T = pred.shape[1]
        iou_all = []
        for t in range(T):
            iou_all.append(compute_iou(pred[:, t].sigmoid().detach().unsqueeze(2), gt[:, t].detach().unsqueeze(1)))
        iou_all = torch.stack(iou_all, 0).mean(0)
    else:
        iou_all = compute_iou(pred.sigmoid().detach().unsqueeze(2), gt.detach().unsqueeze(1))  # [B, N, N]

    matched_pred = torch.zeros_like(pred)
    matched_idx = []

    for b in range(B):
        cost = 1 - iou_all[b].detach().cpu().numpy()
        row_idx, col_idx = linear_sum_assignment(cost)

        reordered = torch.zeros_like(pred[b])
        if pred.ndim == 5:
            reordered[:, col_idx] = pred[b, :, row_idx]
        else:
            reordered[col_idx] = pred[b, row_idx]
        matched_pred[b] = reordered
        matched_idx.append(col_idx)

    return matched_pred, matched_idx


def calculate_metrics(gt_motions, gt_annos, pred_annos_hm, n_frames, num_objs):
    per_seq_obj_ious = []
    per_seq_frame_reds = []
    per_seq_intersect = [0 for _ in range(10)]
    per_seq_num_preds = 0
    per_seq_num_gts = 0
    for frame_idx in range(n_frames):
        gt_motion = gt_motions[frame_idx]
        gt_anno = gt_annos[frame_idx]
        pred_anno_hm = pred_annos_hm[frame_idx]
        per_seq_num_preds += (pred_anno_hm.float().mean(dim=[-1,-2]) > 0).float().sum().item()
        per_seq_num_gts += gt_motion.sum().item()
        obj_ious = compute_iou(pred_anno_hm, gt_anno)
        for thres_idx, thres in enumerate((np.arange(10) * 0.05 + 0.5).tolist()):
            per_seq_intersect[thres_idx] += (obj_ious * gt_motion > thres).float().sum().item()
        per_seq_frame_reds.append(((pred_anno_hm.float().mean(-1).mean(-1) > 0).float() * (1 - gt_motion)).sum())
        per_seq_obj_ious.append(obj_ious)

    # Per-channel motion-IoU: average obj_iou over the motion frames of each channel.
    motion_frames_per_ch = gt_motions.sum(0)                                  # (20,)
    diou_per_ch = (torch.stack(per_seq_obj_ious, 0) * gt_motions).sum(0) / (motion_frames_per_ch + 1e-8)
    # Keep only channels with any motion frames — robust to gaps in GT mask ids
    # (e.g. static-object id sitting between two motion ids). `num_objs` already
    # equals the number of such non-empty channels, so the cardinality is the same.
    motion_channels = (motion_frames_per_ch > 0).nonzero(as_tuple=True)[0]
    j_mov_per_obj = diou_per_ch[motion_channels].tolist()

    mtiou_per_thres = np.array(per_seq_intersect) / (per_seq_num_preds + per_seq_num_gts - np.array(per_seq_intersect) + 1e-8)
    fp_count = torch.stack(per_seq_frame_reds, 0).mean().item()
    return j_mov_per_obj, mtiou_per_thres, fp_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Motion-aware temporal IoU evaluation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, required=True,
                        choices=list(DATASET_TO_CSV_SLICE.keys()),
                        help='Dataset name; selects the merged-CSV slice '
                             '(davis17-im→davis/test, ytvos19-im→ytvos/test).')
    parser.add_argument('--res_dir', type=str, required=True,
                        help='Per-seq prediction directory: <res_dir>/<seq>/00000.png ...')
    parser.add_argument('--anno_dir', type=str, required=True,
                        help='Palette-indexed GT mask root: <anno_dir>/<seq>/00000.png ...')
    parser.add_argument('--time_anno_csv', type=str, required=True,
                        help='Per-seq motion frame ranges. Either a 2-column CSV '
                             '(seq_name,time_anno), or a merged multi-dataset CSV with '
                             'extra dataset/split columns — filtered using --dataset.')
    parser.add_argument('--frame_level_match', action='store_true',
                        help='Also compute + report frame-level Hungarian match (flhm); '
                             'when off, only the sequence-level match (slhm) is shown.')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-sequence results')
    args = parser.parse_args()

    anno_dir = args.anno_dir
    time_anno_csv = args.time_anno_csv
    csv_dataset, csv_split = DATASET_TO_CSV_SLICE[args.dataset]

    time_anno_df = pd.read_csv(time_anno_csv)
    # Merged multi-dataset CSV → filter to this dataset slice (split always 'test').
    if "dataset" in time_anno_df.columns:
        time_anno_df = time_anno_df[time_anno_df["dataset"] == csv_dataset]
    if "split" in time_anno_df.columns:
        time_anno_df = time_anno_df[time_anno_df["split"] == csv_split]
    # Drop rows with empty motion-anno dicts.
    time_anno_df = time_anno_df[time_anno_df["time_anno"].apply(lambda x: bool(ast.literal_eval(x)))].reset_index(drop=True)
    seq_names = time_anno_df["seq_name"].tolist()
    print(f"Dataset: {args.dataset}, {len(seq_names)} sequences")

    # slhm = sequence-level hungarian match (always computed)
    # flhm = frame-level hungarian match (only if --frame_level_match)
    all_j_mov_slhm, all_fp_count_slhm, all_mtiou_slhm = [], [], []
    all_j_mov_flhm, all_fp_count_flhm, all_mtiou_flhm = [], [], []
    per_seq_results = {}

    for seq_name in tqdm(seq_names, desc="mtiou"):
        gt_rgb_annos = np.stack([np.array(Image.open(os.path.join(anno_dir, seq_name, e)))
                                 for e in sorted(os.listdir(os.path.join(anno_dir, seq_name)))], 0)
        H, W = gt_rgb_annos[0].shape
        pred_rgb_annos = np.stack([np.array(Image.open(os.path.join(args.res_dir, seq_name, e)))
                                   for e in sorted(os.listdir(os.path.join(args.res_dir, seq_name)))], 0)
        need_resize = pred_rgb_annos.shape[1:3] != (H, W)
        n_frames = min(gt_rgb_annos.shape[0], pred_rgb_annos.shape[0])
        gt_rgb_annos = gt_rgb_annos[:n_frames]
        pred_rgb_annos = pred_rgb_annos[:n_frames]
        time_anno_dict = ast.literal_eval(time_anno_df[time_anno_df["seq_name"] == seq_name]["time_anno"].values[0])
        gt_motions = obtain_dynamic_labels(time_anno_dict, n_frames=n_frames, n_objs=20).cuda()

        if os.path.exists(os.path.join(args.res_dir, seq_name + "___motion.npy")):
            pred_motions = torch.from_numpy(np.load(os.path.join(args.res_dir, seq_name + "___motion.npy"))).cuda()[..., 0]
        else:
            pred_motions = None

        num_objs = ((torch.from_numpy(np.eye(21, dtype=np.uint8)[gt_rgb_annos].transpose(0, 3, 1, 2)[:, 1:]).cuda()
                     * gt_motions[..., None, None]).mean(dim=[0, -1, -2]) > 0).sum().item()

        gt_annos = []
        pred_annos = []
        pred_annos_flhm = []
        for frame_idx in range(n_frames):
            gt_motion = gt_motions[frame_idx]
            gt_rgb_anno = gt_rgb_annos[frame_idx]
            pred_rgb_anno = pred_rgb_annos[frame_idx]
            gt_anno = torch.tensor((gt_rgb_anno[None] == (np.arange(20)[..., None, None] + 1))).cuda() * gt_motion[:, None, None]
            pred_anno = torch.tensor((pred_rgb_anno[None] == (np.arange(20)[..., None, None] + 1))).float().cuda()
            if need_resize:
                pred_anno = torch.nn.functional.interpolate(pred_anno.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False)[0]
                pred_anno = (pred_anno > 0.5).float()
            if pred_motions is not None:
                pred_motion = (pred_motions[frame_idx] > 0.5).float()
                pred_anno = pred_anno * pred_motion[..., None, None]

            if args.frame_level_match:
                pred_anno_flhm = batch_hungarian_match(pred_anno.unsqueeze(0), gt_anno.unsqueeze(0))[0][0]
                pred_annos_flhm.append(pred_anno_flhm)
            gt_annos.append(gt_anno)
            pred_annos.append(pred_anno)

        gt_annos = torch.stack(gt_annos, 0)
        pred_annos = torch.stack(pred_annos, 0)
        pred_annos_slhm = batch_hungarian_match(pred_annos.unsqueeze(0), gt_annos.unsqueeze(0))[0][0]

        j_mov_slhm, mtiou_slhm, fp_count_slhm = calculate_metrics(gt_motions, gt_annos, pred_annos_slhm, n_frames, num_objs)
        all_j_mov_slhm.extend(j_mov_slhm)
        all_mtiou_slhm.append(mtiou_slhm)
        all_fp_count_slhm.append(fp_count_slhm)

        if args.frame_level_match:
            pred_annos_flhm = torch.stack(pred_annos_flhm, 0)
            j_mov_flhm, mtiou_flhm, fp_count_flhm = calculate_metrics(gt_motions, gt_annos, pred_annos_flhm, n_frames, num_objs)
            all_j_mov_flhm.extend(j_mov_flhm)
            all_mtiou_flhm.append(mtiou_flhm)
            all_fp_count_flhm.append(fp_count_flhm)

        if args.verbose:
            per_seq_results[seq_name] = {
                "j_mov_slhm": float(np.mean(j_mov_slhm)) if j_mov_slhm else float('nan'),
                "mtiou_slhm": float(np.mean(mtiou_slhm)),
            }
            if args.frame_level_match:
                per_seq_results[seq_name].update({
                    "j_mov_flhm": float(np.mean(j_mov_flhm)) if j_mov_flhm else float('nan'),
                    "mtiou_flhm": float(np.mean(mtiou_flhm)),
                })

    mtiou_slhm_mean = float(np.mean(np.stack(all_mtiou_slhm, 0).mean(0)))
    j_mov_slhm_mean = float(np.mean(all_j_mov_slhm))
    fp_count_slhm_mean = float(np.mean(all_fp_count_slhm))

    if args.frame_level_match:
        mtiou_flhm_mean = float(np.mean(np.stack(all_mtiou_flhm, 0).mean(0)))
        j_mov_flhm_mean = float(np.mean(all_j_mov_flhm))
        fp_count_flhm_mean = float(np.mean(all_fp_count_flhm))

        print("=== Sequence-level Hungarian match ===")
        print(f"  mtiou: {mtiou_slhm_mean:.4f}  j_mov: {j_mov_slhm_mean:.4f}  fp_count: {fp_count_slhm_mean:.4f}")
        print("=== Frame-level Hungarian match ===")
        print(f"  mtiou: {mtiou_flhm_mean:.4f}  j_mov: {j_mov_flhm_mean:.4f}  fp_count: {fp_count_flhm_mean:.4f}")
    else:
        print(f"mtiou: {mtiou_slhm_mean:.4f}  j_mov: {j_mov_slhm_mean:.4f}  fp_count: {fp_count_slhm_mean:.4f}")

    if args.verbose:
        print("\nPer-sequence results (sequence-level Hungarian match):")
        for seq_name, r in per_seq_results.items():
            print(f"  {seq_name:30s}  j_mov: {r['j_mov_slhm']:.4f}  mtiou: {r['mtiou_slhm']:.4f}")
        if args.frame_level_match:
            print("\nPer-sequence results (frame-level Hungarian match):")
            for seq_name, r in per_seq_results.items():
                print(f"  {seq_name:30s}  j_mov: {r['j_mov_flhm']:.4f}  mtiou: {r['mtiou_flhm']:.4f}")
