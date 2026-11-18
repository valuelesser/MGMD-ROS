"""Build inverse image warps from timestamped body orientations on CUDA."""

import torch
from torch.nn.functional import normalize


def camera_warps(frame_ns, imu_ns, imu_xyzw, intrinsic, body_to_camera):
    """Return the center-to-previous and center-to-next warps, plus their validity."""
    upper = torch.searchsorted(imu_ns, frame_ns, right=True)
    right = upper.clamp(1, imu_ns.numel() - 1)
    left = right - 1
    span = imu_ns[right] - imu_ns[left]
    covered = (upper > 0) & (upper < imu_ns.numel()) & (span > 0) & (span <= 50_000_000)
    q0, q1 = imu_xyzw[left], imu_xyzw[right]
    for q in (q0, q1):
        covered &= torch.isfinite(q).all(dim=1) & ((q.norm(dim=1) - 1).abs() < .02)
    q0, q1 = normalize(q0), normalize(q1)
    dot = (q0 * q1).sum(dim=1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(max=1)
    fraction = ((frame_ns - imu_ns[left]).to(q0.dtype) / span.clamp_min(1)).clamp(0, 1)[:, None]
    theta = dot.clamp(max=1 - 1e-7).acos()
    spherical = (torch.sin((1 - fraction) * theta) * q0 + torch.sin(fraction * theta) * q1) / theta.sin()
    x, y, z, w = normalize(torch.where(dot > .9995, q0.lerp(q1, fraction), spherical)).unbind(1)
    rotations = torch.stack((1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
                             2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
                             2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)), dim=1).reshape(-1, 3, 3)
    center = rotations[1:-1]
    rays = body_to_camera.T @ torch.linalg.inv_ex(intrinsic)[0]
    projection = intrinsic @ body_to_camera
    warps = torch.stack([projection @ neighbor.transpose(1, 2) @ center @ rays
                         for neighbor in (rotations[:-2], rotations[2:])], dim=1)
    return warps, covered[:-2] & covered[1:-1] & covered[2:]
