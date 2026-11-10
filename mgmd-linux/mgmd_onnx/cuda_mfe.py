"""Device image operations for the MFE branch with supplied alignment."""

import torch
from torch.nn.functional import avg_pool2d, grid_sample, max_pool2d, pad


def bgr_to_gray(image):
    if not image.is_cuda or image.dtype != torch.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a CUDA uint8 HWC BGR image")
    b, g, r = image.int().unbind(dim=2)
    return ((3735 * b + 19235 * g + 9798 * r + 16384) >> 15).to(torch.uint8)


def warp_inverse(source, matrix):
    """Apply a destination-to-source homography with bilinear sampling."""
    h, w = source.shape
    x = torch.arange(w, dtype=torch.float64, device=source.device)[None, :]
    y = torch.arange(h, dtype=torch.float64, device=source.device)[:, None]
    z = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    factor = torch.where(z != 0, 1 / z, 0)
    sx = (matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) * factor
    sy = (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) * factor
    grid = torch.stack(((2 * sx + 1) / w - 1, (2 * sy + 1) / h - 1), dim=-1)
    warped = grid_sample(source[None, None].double(), grid[None], padding_mode="zeros",
                         align_corners=False)
    return warped[0, 0].round().clamp_(0, 255).to(torch.uint8)


def threshold_direction(center, aligned, valid, cfg):
    delta = (center.to(torch.int16) - aligned).abs()
    threshold = cfg.threshold
    if cfg.adaptive_thresh:
        threshold = threshold + torch.div(delta.sum(dtype=torch.int64), delta.numel(),
                                          rounding_mode="floor")
    valid = valid.bool()
    mask = (delta > threshold) & valid
    if cfg.median_blur >= 3:
        radius = cfg.median_blur // 2
        padded = pad(mask[None, None].float(), (radius,) * 4, mode="replicate")
        mask = (avg_pool2d(padded, cfg.median_blur, stride=1)[0, 0] > .5) & valid
    return mask, threshold


def combine_motion(previous, following, mode):
    if following is None or mode == "2frame":
        return previous
    if mode == "3frame_and":
        return previous & following
    return previous | following


def morphology(mask, size, open_iters, close_iters):
    value = mask[None, None].float()
    # For binary data max pooling is dilation. Negation gives erosion with a neutral border.
    for iterations, signs in ((open_iters, (-1, 1)), (close_iters, (1, -1))):
        for sign in signs:
            for _ in range(iterations):
                value = sign * max_pool2d(sign * value, size, stride=1, padding=size // 2)
    return value[0, 0].bool()
