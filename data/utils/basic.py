"""Tensor padding + annotation-mask refinement helpers."""
import torch
import torch.nn.functional as F


def pad_axis(tensor, dim, target_length, pad_value=0):
    """
    Pads a specific axis of a PyTorch tensor to a target length.

    Args:
        tensor (torch.Tensor): The input tensor.
        dim (int): The dimension/axis to pad.
        target_length (int): The desired length of that dimension.
        pad_value (float): The value to fill the padding with.

    Returns:
        torch.Tensor: The padded tensor.
    """
    # 1. Get current length of the specified dimension
    current_length = tensor.shape[dim]

    # 2. Calculate how much padding is needed
    pad_amount = target_length - current_length

    # If the tensor is already long enough, return it as is
    if pad_amount <= 0:
        return tensor

    # 3. Construct the 'pad' tuple for F.pad
    # F.pad expects inputs in reverse order: (last_dim_left, last_dim_right, 2nd_last_left, ...)
    ndim = tensor.ndim

    # Handle negative indexing (e.g., dim=-1)
    if dim < 0:
        dim += ndim

    # Initialize a list of zeros (2 values per dimension)
    pad_config = [0] * (2 * ndim)

    # Calculate the index in the pad_config list corresponding to the 'right' side of 'dim'
    # Formula: (distance from last dim) * 2 + 1 (for right side)
    pad_idx = (ndim - dim - 1) * 2 + 1

    # Set the padding amount
    pad_config[pad_idx] = pad_amount

    # 4. Apply padding
    # We convert the list to a tuple as required by F.pad
    return F.pad(tensor, tuple(pad_config), value=pad_value)


def refine_anno(anno_tensor, dynamic_anno_tensor):
    """
    Given segmentation annotations and dynamic annotations, refine them according to:
    (i) if dynamic annotation is 0, set segmentation annotation all to 0;
    (ii) if segmentation annotation sum is 0, set dynamic annotation to 0.

    Args:
        anno_tensor (torch.Tensor): N H W
        dynamic_anno_tensor (torch.Tensor): H W

    Returns:
        refined_anno_tensor (torch.Tensor): N H W
        refined_dynamic_anno_tensor (torch.Tensor): H W
    """
    refined_dynamic_anno_tensor = (anno_tensor.mean(dim=[-2, -1]) > 0).to(torch.float32) * dynamic_anno_tensor
    refined_anno_tensor = anno_tensor * dynamic_anno_tensor.unsqueeze(1).unsqueeze(2)
    return refined_anno_tensor, refined_dynamic_anno_tensor
