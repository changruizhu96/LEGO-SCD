from __future__ import annotations
import torch
from gsedit_scd.vendor.oscd.utils.sh_utils import eval_sh

@torch.no_grad()
def view_colors(gaussians, view):
    sh = gaussians.get_features.transpose(1, 2).reshape(
        -1, 3, (gaussians.max_sh_degree + 1) ** 2
    )
    direction = gaussians.get_xyz - view.camera_center.reshape(1, 3)
    direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return (eval_sh(gaussians.active_sh_degree, sh, direction) + 0.5).clamp(
        0.0, 1.0
    )

