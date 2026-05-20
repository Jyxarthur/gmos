"""GMOS_S inference: foreground/background segmentation per video.

Per-video pipeline:
  1. Pi3 forward (5-frame window per dataset sample, with DINOv2 caching across windows).
  2. SAM2 image encoder forward on mid frame.
  3. GMOS_S fusion → single B×1×H×W fg mask.
  4. Save per-frame indexed PNGs and/or overlay MP4.

CLI surface:
  --pi3_ckpt, --sam2_ckpt, --batch_size, --gmos_s_ckpt,
  --input_rgb, --save_mode {prob, mask, mp4, +combos}, --stride, --offload_cpu
"""
import os
import sys

# Local sam2 at core/sam2
_local_sam2 = os.path.join(os.path.dirname(__file__), "core", "sam2")
if _local_sam2 not in sys.path:
    sys.path.insert(0, _local_sam2)

from data.loader.video_loader import video_dataset    # noqa: E402
from core.sam2_encoder import SAM2Encoder             # noqa: E402
from core.pi3_encoder import Pi3Encoder               # noqa: E402
from core.gmos_s import GMOS_S                        # noqa: E402
from utils import (                                   # noqa: E402
    encode_pi3_with_cache, resolve_fps_stride,
    parse_input_rgb, LoadedImageDirVideo,
)
from argparse import ArgumentParser                   # noqa: E402
import torch                                          # noqa: E402
import numpy as np                                    # noqa: E402
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import mediapy as media
from PIL import Image

from prop.prop_utils import save_indexed_png, overlay_masks


def inference(args, pi3_encoder, sam_encoder, gmos, val_loader, device,
              save_mode="mask", fps=30.0, stride=2, use_dino_cache=True, offload_cpu=False,
              frames_array=None):
    gmos.eval()

    all_seq_names = []
    all_mid_indices = []
    all_annos = []
    all_probs = []
    pi3_cache = {}

    for idx, info in tqdm(enumerate(val_loader), total=len(val_loader), desc="inference_gmos_s"):
        sam_rgb = info["sam_rgb"].to(device)
        annos = info["annos"].to(device)                           # B N H W
        seq_names = info["seq_name"]
        mid_indices = info["mid_index"]
        video_indices = info["video_indices"].to(device)           # (B, N)
        _, _, H, W = annos.shape
        if "ori_hw" in info:
            ori_H, ori_W = info["ori_hw"][0, 0].item(), info["ori_hw"][0, 1].item()
        else:
            ori_H, ori_W = H, W
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
                backbone_out = sam_encoder.forward_image(sam_rgb)
                _, vision_feats, _, _ = sam_encoder._prepare_backbone_features(backbone_out)
                sam_btn, sam_highres = sam_encoder._get_vision_features(vision_feats)
                sam_btn = sam_btn.detach().clone().float()
                sam_highres = [feat.detach().clone().float() for feat in sam_highres]

            with torch.inference_mode():
                predictions = gmos(pi3_hidden_mid, sam_btn, sam_highres)

                fg_mask = predictions["fg_mask"]                   # B 1 H' W'
                fg_mask = F.interpolate(fg_mask, size=(ori_H, ori_W), mode="bilinear", align_corners=False)

        all_seq_names.extend(seq_names)
        all_mid_indices.extend(mid_indices.detach().cpu().tolist())
        fg_prob = fg_mask.sigmoid()
        fg_pred = (fg_prob > 0.5).float()
        all_annos.append(fg_pred[:, 0].detach().cpu().numpy().astype(np.int32))
        all_probs.append(fg_prob[:, 0].detach().cpu().numpy().astype(np.float32))

    all_seq_names = np.array(all_seq_names)
    all_annos = np.concatenate(all_annos, axis=0)              # Total H W (int, 0 or 1)
    all_probs = np.concatenate(all_probs, axis=0)              # Total H W (float, [0,1])

    for seq_name in np.unique(all_seq_names).tolist():
        seq_indices = np.where(all_seq_names == seq_name)[0]
        seq_annos = all_annos[seq_indices]                     # T H W (int)
        seq_probs = all_probs[seq_indices]                     # T H W (float)

        if "mask" in save_mode:
            save_dir = os.path.join(args.save_base_dir, "frame", seq_name)
            os.makedirs(save_dir, exist_ok=True)
            for fi in range(seq_annos.shape[0]):
                save_indexed_png(seq_annos[fi].astype(np.uint8),
                                 os.path.join(save_dir, str(fi).zfill(5) + ".png"))

        if "prob" in save_mode:
            save_dir = os.path.join(args.save_base_dir, "prob", seq_name)
            os.makedirs(save_dir, exist_ok=True)
            prob_u8 = np.clip(seq_probs * 255.0, 0, 255).astype(np.uint8)
            for fi in range(prob_u8.shape[0]):
                Image.fromarray(prob_u8[fi], mode="L").save(
                    os.path.join(save_dir, str(fi).zfill(5) + ".png"))

        if "mp4" in save_mode:
            assert frames_array is not None, "frames_array required for mp4 save"
            save_video_dir = os.path.join(args.save_base_dir, "mp4")
            os.makedirs(save_video_dir, exist_ok=True)
            overlay = overlay_masks(np.asarray(frames_array), seq_annos.astype(np.uint8), alpha=0.5)
            media.write_video(
                os.path.join(save_video_dir, f"{seq_name}.mp4"),
                overlay, fps=fps,
            )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--pi3_ckpt', type=str, required=True,
                        help="Path to the Pi3 checkpoint (.safetensors or .pth).")
    parser.add_argument('--sam2_ckpt', type=str, required=True,
                        help="Path to the SAM2 Hiera-L checkpoint (.pt).")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gmos_s_ckpt', type=str, required=True,
                        help="Path to the trained GMOS-S checkpoint (.pth).")
    parser.add_argument('--save_dir', type=str, required=True,
                        help="Output directory. Subdirs created inside: "
                             "{prob, frame, mp4}/<seq>/...")
    parser.add_argument('--input_rgb', type=str, nargs='+', required=True,
                        help="One or more inputs. Accepts: (i) a single .mp4 file, "
                             "(ii) multiple .mp4 files, (iii) a directory of JPEG/PNG frames, "
                             "(iv) multiple directories of JPEG/PNG frames. All entries must be "
                             "the same kind (all .mp4 or all image dirs). For image-dir inputs, "
                             "--fps or --stride must be specified explicitly.")
    parser.add_argument('--save_mode', type=str, default="prob",
                        choices=["prob", "mask", "mp4",
                                 "prob+mask", "prob+mp4", "mask+mp4",
                                 "prob+mask+mp4"],
                        help="What to write (combine with '+'): "
                             "'prob' = per-frame 8-bit grayscale PNGs of sigmoid(fg_logit) ∈ [0,1] "
                             "(0=bg, 255=fg), at video resolution; "
                             "'mask' = per-frame indexed PNGs (DAVIS palette, video resolution); "
                             "'mp4' = single MP4 per video with 0.5-alpha DAVIS-palette overlay "
                             "at original video resolution + FPS. Default 'prob'.")
    parser.add_argument('--stride', type=int, default=None,
                        help="Frame stride between window elements in the proposer's 5-frame Pi3 windows. "
                             "If unset, derived from --fps (or the video's metadata fps) via "
                             "stride = max(1, int(fps/8)). If both --stride and --fps are set, "
                             "--stride wins.")
    parser.add_argument('--fps', type=float, default=None,
                        help="Override the per-video fps used to compute --stride. Ignored if "
                             "--stride is also set. If neither is set, fps is read from the video "
                             "metadata per file. Formula: stride = max(1, int(fps/8)).")
    parser.add_argument('--offload_cpu', action='store_true',
                        help="Offload the DINOv2 cache to pinned CPU memory; "
                             "use for long videos when GPU memory is tight.")
    args = parser.parse_args()
    device = torch.device('cuda')

    # --- Load Pi3 encoder ---
    pi3_encoder = Pi3Encoder().to(device).eval()
    if args.pi3_ckpt.endswith('.safetensors'):
        from safetensors.torch import load_file
        weight = load_file(args.pi3_ckpt)
    else:
        weight = torch.load(args.pi3_ckpt, map_location=device, weights_only=False)
    pi3_encoder.load_state_dict(weight, strict=False)
    pi3_encoder.eval()
    for param in pi3_encoder.parameters():
        param.requires_grad = False

    # --- Load SAM2 encoder ---
    sam_encoder = SAM2Encoder().to(device)
    ckpt = torch.load(args.sam2_ckpt, map_location=device)
    state_dict = ckpt.get("model", ckpt)
    filtered = {k: v for k, v in state_dict.items()
                if "image_encoder" in k or "sam_mask_decoder.conv_s0" in k
                or "sam_mask_decoder.conv_s1" in k or "no_mem_embed" in k}
    sam_encoder.load_state_dict(filtered, strict=False)
    sam_encoder.eval()
    for param in sam_encoder.parameters():
        param.requires_grad = False

    # --- Load GMOS_S model (settings live in GMOS_S.__init__ defaults) ---
    gmos = GMOS_S().to(device)

    print("Loading ckpt")
    state_dict = torch.load(args.gmos_s_ckpt)['model_state_dict']
    if any(k.startswith("module.") for k in state_dict.keys()):
        from collections import OrderedDict
        state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    missing, unexpected = gmos.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys: {missing}")

    args.save_base_dir = args.save_dir
    os.makedirs(args.save_base_dir, exist_ok=True)
    print(f"save_base_dir = {args.save_base_dir}  (save_mode={args.save_mode})")

    # --- Resolve --input_rgb ---
    kind, entries = parse_input_rgb(args.input_rgb)
    print(f"--input_rgb: kind={kind}, n={len(entries)}")

    for path, seq_name in entries:
        stride, fps_used, source = resolve_fps_stride(path, args.fps, args.stride, kind=kind)
        print(f"[{seq_name}] fps_source={source} fps={fps_used} stride={stride}")

        if kind == "imgdir":
            output_fps = args.fps if args.fps is not None else 24.0
            video = LoadedImageDirVideo(path, fps=output_fps)
            dataset = video_dataset(
                video_path=None, video=video, seq_name=seq_name,
                strides=[stride], num_objs=20, sam_version="v2",
            )
            fps = output_fps
        else:
            dataset = video_dataset(
                video_path=path, strides=[stride], num_objs=20, sam_version="v2",
            )
            fps = float(dataset.video.metadata.fps) if "mp4" in args.save_mode else 30.0
        val_loader = DataLoader(dataset, num_workers=8, batch_size=args.batch_size,
                                shuffle=False, pin_memory=True, drop_last=False)
        frames_array = dataset.video if "mp4" in args.save_mode else None
        inference(args, pi3_encoder, sam_encoder, gmos, val_loader, device,
                  save_mode=args.save_mode, fps=fps, stride=stride,
                  use_dino_cache=True, offload_cpu=args.offload_cpu,
                  frames_array=frames_array)
