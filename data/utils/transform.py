"""Frame-window helpers + PIL → Pi3 tensor preprocessing used by the loader."""
import os
import math
import cv2
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
from scipy.interpolate import CubicSpline



def group_list_with_stride(data, window_size=5, stride=1, pattern=None):
    n = len(data)
    if pattern is None:
        pattern = [1] * (window_size - 1)
    scaled_pattern = [p * stride for p in pattern]
    # Build offsets from center: cumsum from center outward in both directions
    mid = (window_size - 1) // 2
    offsets = [0] * window_size
    for j in range(mid - 1, -1, -1):
        offsets[j] = offsets[j + 1] - scaled_pattern[j]
    for j in range(mid + 1, window_size):
        offsets[j] = offsets[j - 1] + scaled_pattern[j - 1]
    result = []
    for i in range(n):
        indices = [max(0, min(n - 1, i + o)) for o in offsets]
        group = [data[idx] for idx in indices]
        result.append(group)
    return result

def group_list_with_stride_no_padding(data, window_size=5, stride=1):
    n = len(data)
    offset = (window_size - 1) * stride // 2
    result = []
    for i in range(offset, n - offset):
        indices = [i + (j * stride - offset) for j in range(window_size)]
        result.append([data[idx] for idx in indices])
    return result

def index_to_binary_masks(idx_map, n_mask=None):
    if n_mask is None:
        N = idx_map.max() + 1
    else:
        N = n_mask + 1
    H, W = idx_map.shape
    if N == 1:
        masks = np.zeros((1, H, W), dtype=np.float32)
    else:
        masks = np.zeros((N, H, W), dtype=np.float32)
        for i in range(N):
            masks[i] = (idx_map == i)
        masks = masks[1:]
    return masks


def obtain_dynamic_labels(data_dict, indices, n_obj=None):
    idx_tensor = torch.tensor(indices).unsqueeze(1)
    max_key = max((int(k) for k in data_dict), default=0)
    num_cols = max(max_key, n_obj or 0)
    result = torch.zeros(len(indices), num_cols)

    for key, ranges_list in data_dict.items():
        col_idx = int(key) - 1  # key '1' -> index 0, key '4' -> index 3
        ranges = torch.tensor(ranges_list)
        if ranges.numel() == 0:
            continue
        matches = (idx_tensor >= ranges[:, 0]) & (idx_tensor <= ranges[:, 1])
        result[:, col_idx] = matches.any(dim=1).float()

    return result


def pil_image_to_tensor_pi3(sources, interval=1, PIXEL_LIMIT=255000):
    first_img = sources[0]
    W_orig, H_orig = first_img.size
    scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
    W_target, H_target = W_orig * scale, H_orig * scale
    k, m = round(W_target / 14), round(H_target / 14)
    while (k * 14) * (m * 14) > PIXEL_LIMIT:
        if k / m > W_target / H_target: k -= 1
        else: m -= 1
    TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
    # print(f"All images will be resized to a uniform size: ({TARGET_W}, {TARGET_H})")

    tensor_list = []
    # Define a transform to convert a PIL Image to a CxHxW tensor and normalize to [0,1]
    to_tensor_transform = transforms.ToTensor()
    
    for img_pil in sources:
        try:
            # Resize to the uniform target size
            resized_img = img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            # Convert to tensor
            img_tensor = to_tensor_transform(resized_img)
            tensor_list.append(img_tensor)
        except Exception as e:
            print(f"Error processing an image: {e}")

    if not tensor_list:
        print("No images were successfully processed.")
        return torch.empty(0)

    return torch.stack(tensor_list, dim=0)



def get_virtual_camera_homography(w, h, pitch, yaw, roll, zoom_factor=1.0):
    """
    Computes the Homography matrix that applies 3D Rotation + Zoom.
    """
    # 1. Define Camera Intrinsics (K)
    # Arbitrary Field of View (FOV) of 50 degrees is standard for realistic look
    fov = 50 
    f = (w / 2) / np.tan(np.deg2rad(fov / 2))
    cx, cy = w / 2, h / 2
    
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
    K_inv = np.linalg.inv(K)

    # 2. Rotation Matrix (R)
    rx, ry, rz = np.deg2rad(pitch), np.deg2rad(yaw), np.deg2rad(roll)
    
    # Standard 3D Rotation Matrices
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    
    # Combined Rotation
    R = Rz @ Ry @ Rx

    # 3. Zoom Matrix (S)
    # We scale relative to the center (cx, cy)
    # To zoom IN, we scale the image UP (zoom_factor > 1)
    scale_mat = np.array([
        [zoom_factor, 0, (1 - zoom_factor) * cx],
        [0, zoom_factor, (1 - zoom_factor) * cy],
        [0, 0, 1]
    ])

    # 4. Total Homography
    # The order: Project 3D -> Rotate -> Project Back -> Zoom
    # Note: We apply scale LAST in the chain (mathematically left-multiplied)
    # H = Scale * (K * R * K_inv)
    H_rot = K @ R @ K_inv
    H_final = scale_mat @ H_rot
    
    return H_final

def calculate_max_safe_angles(w, h, zoom_factor):
    """
    Estimates the maximum rotation angle allowed before seeing black borders.
    """
    # The margin is the number of pixels we have "spare" on the sides
    # If zoom is 1.2, we are using w/1.2 pixels. The margin is roughly (w - w/1.2) / 2
    margin_x = (w - (w / zoom_factor)) / 2
    margin_y = (h - (h / zoom_factor)) / 2
    
    # Focal length (approximate based on 50 deg FOV)
    f = (w / 2) / np.tan(np.deg2rad(50 / 2))
    
    # The shift in pixels due to rotation theta is approx: delta = f * tan(theta)
    # So max_tan_theta = margin / f
    max_pitch = np.rad2deg(np.arctan(margin_y / f))
    max_yaw   = np.rad2deg(np.arctan(margin_x / f))
    
    # Return 90% of the theoretical max to be safe
    return max_pitch * 0.75, max_yaw * 0.75

def generate_safe_camera_sequence(image, num_frames=5, zoom=1.2):
    h, w = image.shape[:2]

    # --- 1. Calculate Limits ---
    # We determine how much we can rotate without going out of bounds
    limit_p, limit_y = calculate_max_safe_angles(w, h, zoom)
    if limit_p == 0:
        limit_r = 0.0
    else:
        limit_r = 2.0 # Roll is usually safe in small amounts
    
    # print(f"Zoom: {zoom}x. Safe angle limits -> Pitch: +/-{limit_p:.1f}°, Yaw: +/-{limit_y:.1f}°")

    # --- 2. Generate Smooth Paths ---
    # We create random waypoints within these safe limits
    key_times = [0, 0.5, 1.0] # Start, Middle, End
    
    # Start at 0 (centered), go to random mid, end at random dest
    p_way = [0, np.random.uniform(-limit_p, limit_p), np.random.uniform(-limit_p, limit_p)]
    y_way = [0, np.random.uniform(-limit_y, limit_y), np.random.uniform(-limit_y, limit_y)]
    r_way = [0, np.random.uniform(-limit_r, limit_r), np.random.uniform(-limit_r, limit_r)]
    
    # Create Splines
    sp_p = CubicSpline(key_times, p_way)
    sp_y = CubicSpline(key_times, y_way)
    sp_r = CubicSpline(key_times, r_way)
    
    # Generate frame timestamps
    times = np.linspace(0, 0.5, num_frames)

    # print(f"Generating {num_frames} frames (Resolution: {w}x{h})...")

    # --- 3. Render Loop ---
    final_images = []
    for i in range(num_frames):
        t = times[i]
        p, y, r = sp_p(t), sp_y(t), sp_r(t)
        
        # Calculate Homography
        H = get_virtual_camera_homography(w, h, p, y, r, zoom_factor=zoom)
        
        # Warp
        # We use dsize=(w, h) so the output is the SAME RESOLUTION as input.
        # The 'zoom' matrix we applied handles the scaling logic.
        frame = cv2.warpPerspective(image, H, (w, h), 
                                    flags=cv2.INTER_CUBIC, # High quality resizing
                                    borderMode=cv2.BORDER_CONSTANT, 
                                    borderValue=(0, 0, 0))
        final_images.append(frame)

    final_images = np.stack(final_images, 0)
    return final_images