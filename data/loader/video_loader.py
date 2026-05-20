"""Per-video dataset for inference.

Yields stride-`s` windows of `num_frames` consecutive frames. Each window
returns the Pi3 input tensor (T,C,H,W), the SAM2 input tensor for the
middle frame, and per-frame metadata. Backed by `mediapy.read_video` for
mp4 inputs and by a caller-supplied object (see `LoadedImageDirVideo`
in `utils.py`) for image-directory inputs.
"""
import os

import mediapy as media
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

import data.utils as data_utils
from core.sam2.sam2.utils.transforms import SAM2Transforms


class video_dataset(Dataset):
    def __init__(
        self,
        video_path=None,
        strides = [1],
        num_frames = 5,
        num_objs = 20,
        sam_version = "v2",
        square_input=False,
        pattern=None,
        video=None,
        seq_name=None,
    ):
        """Either pass `video_path` (an mp4 path; this loader will decode via mediapy)
        or pass an already-loaded `video` object exposing `.shape` + `__getitem__`
        (e.g. LoadedImageDirVideo). `seq_name` overrides the auto-derived name."""
        self.video_path = video_path
        self.num_frames = num_frames  # Only odd numbers
        self.num_objs = num_objs
        self.sam_version = sam_version
        self.square_input = square_input

        if self.sam_version == "v2":
            self._sam2_transforms = SAM2Transforms(
                resolution=1024,
                mask_threshold=0.,
                max_hole_area=0.,
                max_sprinkle_area=0.,
            )
        elif self.sam_version == "v3":
            self._sam3_transforms = v2.Compose(
                [
                    v2.ToDtype(torch.uint8, scale=True),
                    v2.Resize(size=(1008, 1008)),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
        else:
            raise NotImplementedError

        self.samples = []
        if video is not None:
            self.video = video
        else:
            self.video = media.read_video(self.video_path)
        if seq_name is not None:
            self._seq_name = seq_name
        elif video_path is not None:
            self._seq_name = os.path.basename(video_path).replace(".mp4", "")
        else:
            raise ValueError("Must provide either video_path or seq_name")
        frame_list = np.arange(self.video.shape[0]).tolist()
        for stride in strides:
            example_list = data_utils.group_list_with_stride(frame_list, window_size=self.num_frames, stride=stride, pattern=pattern)
            self.samples.extend(example_list)

       
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_indices = self.samples[idx]
        mid_index = video_indices[self.num_frames // 2]
        seq_name = self._seq_name
        imgs = self.video[video_indices]
        rgb_images = [Image.fromarray(frame) for frame in imgs]
        if self.sam_version == "v2":
            sam_rgb_tensor = self._sam2_transforms(rgb_images[self.num_frames // 2])
        elif self.sam_version == "v3":
            image = v2.functional.to_image(rgb_images[self.num_frames // 2])
            sam_rgb_tensor = self._sam3_transforms(image)
        else:
            raise NotImplementedError
        pi3_rgb_tensors = data_utils.pil_image_to_tensor_pi3(rgb_images)

        if self.square_input:
            H, W = 512, 512
        else:
            H, W = pi3_rgb_tensors.shape[-2:]

        anno_label = np.zeros((self.num_objs, H, W))
        anno_tensor = torch.zeros(self.num_objs, H, W)
        dynamic_anno_tensor = torch.zeros(self.num_objs)
        dynamic_gt_indices = torch.zeros(self.num_objs)
        ori_H, ori_W = self.video.shape[1], self.video.shape[2]
        info = {
            "pi3_rgbs": pi3_rgb_tensors,                          # T, C, H, W
            "sam_rgb": sam_rgb_tensor,                            # C, H, W
            "annos": anno_tensor.to(torch.uint8),                         # N, H, W
            "dynamic_labels": dynamic_anno_tensor.long(),         # N
            "dynamic_gt_indices": dynamic_gt_indices.long(),      # N
            "seq_name": seq_name,
            "mid_index": mid_index,
            "video_indices": torch.tensor(video_indices),         # T (per-frame indices in the window)
            "ori_hw": torch.tensor([ori_H, ori_W]),               # 2
        }
        return info

