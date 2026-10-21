from __future__ import annotations
import torch
import torch.nn.functional as F
from gsedit_scd.vendor.oscd.gaussian_renderer import render
from gsedit_scd.vendor.oscd.utils.general_utils import build_rotation
from gsedit_scd.vendor.oscd.utils.sh_utils import eval_sh

def canonical_normals(gaussians, camera_positions):
    scaling = gaussians.get_scaling
    rotation = build_rotation(gaussians.get_rotation)
    smallest_axis = scaling.argmin(dim=1)
    rows = torch.arange(scaling.shape[0], device=scaling.device)
    normals = rotation[rows, :, smallest_axis]
    mean_camera = torch.as_tensor(
        camera_positions,
        device=normals.device,
        dtype=normals.dtype,
    ).mean(dim=0)
    direction = mean_camera[None] - gaussians.get_xyz
    sign = torch.where(
        (normals * direction).sum(dim=1, keepdim=True) >= 0.0, 1.0, -1.0
    )
    return (normals * sign).contiguous()


def reliable_object_gaussians(latent_map, count, device):
    selected = torch.zeros(count, dtype=torch.bool, device=device)
    accepted = 0
    for component in latent_map.get("components", []):
        if int(component.get("view_count", 0)) < 3:
            continue
        if float(component.get("mean_loo_iou", 0.0)) < 0.5:
            continue
        gaussian_ids = component.get("gaussian_ids")
        if gaussian_ids is None:
            continue
        gaussian_ids = torch.as_tensor(
            gaussian_ids, device=device, dtype=torch.long
        )
        gaussian_ids = gaussian_ids[(gaussian_ids >= 0) & (gaussian_ids < count)]
        selected[gaussian_ids] = True
        accepted += 1
    if accepted == 0:
        raise RuntimeError("Latent-object map contains no reliable components")
    return selected, accepted


def view_colors(gaussians, view):
    sh = gaussians.get_features.transpose(1, 2).reshape(
        -1, 3, (gaussians.max_sh_degree + 1) ** 2
    )
    direction = gaussians.get_xyz - view.camera_center.reshape(1, 3)
    direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return (eval_sh(gaussians.active_sh_degree, sh, direction) + 0.5).clamp(
        0.0, 1.0
    )


@torch.no_grad()
def render_relighting_basis(
    view, gaussians, pipe, background, normals, object_gaussians,
    return_ray_stats=False, normal_conditioned=True,
):
    colors = view_colors(gaussians, view).contiguous()
    base_package = render(
        view,
        gaussians,
        pipe,
        background,
        override_color=colors,
        return_ray_stats=True,
        return_owner_ids=True,
    )
    ray_stats = base_package["ray_stats"]
    if ray_stats.ndim != 3 or ray_stats.shape[0] < 9:
        raise RuntimeError("Relighting mixture null requires 9-channel ray statistics")
    coverage = ray_stats[0].detach().clamp(0.0, 1.0)
    owner_ids = ray_stats[7:9].round().long() - 1
    object_pixels = torch.zeros_like(coverage, dtype=torch.bool)
    for slot in range(owner_ids.shape[0]):
        owners = owner_ids[slot]
        valid = (owners >= 0) & (owners < object_gaussians.numel())
        slot_object = torch.zeros_like(valid)
        slot_object[valid] = object_gaussians[owners[valid]]
        object_pixels |= slot_object & (ray_stats[1 + slot] > 1e-3)

    basis = []
    for axis in range(3) if normal_conditioned else ():
        encoded = colors * (0.5 + 0.5 * normals[:, axis : axis + 1])
        encoded_render = render(
            view,
            gaussians,
            pipe,
            background,
            override_color=encoded.contiguous(),
        )["render"]
        basis.append(2.0 * encoded_render - base_package["render"])
    result = (
        base_package["render"].detach(), coverage, torch.stack(basis) if basis else None, object_pixels
    )
    if return_ray_stats:
        return result + (ray_stats.detach(),)
    return result


def nuisance_primitive_colors(colors, normals, coefficient):
    transformed = coefficient[:3] * colors + coefficient[3:6]
    if coefficient.numel() == 9:
        directional = torch.einsum("nd,d->n", normals, coefficient[6:9])
        transformed = transformed + colors * directional[:, None]
    return transformed


def _resize(value, size, mode="bilinear"):
    value = value[None, None] if value.ndim == 2 else value[None]
    if mode == "nearest":
        return F.interpolate(value.float(), size=size, mode=mode)[0]
    return F.interpolate(
        value.float(), size=size, mode=mode, align_corners=False
    )[0]


def _design_matrix(base, coverage, basis, pixel_mask, normal_conditioned):
    pixel_indices = pixel_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
    base_pixels = base.reshape(3, -1)[:, pixel_indices].T
    coverage_pixels = coverage.reshape(-1)[pixel_indices]
    rows = pixel_indices.numel() * 3
    columns = 9 if normal_conditioned else 6
    design = torch.zeros(rows, columns, device=base.device, dtype=torch.float32)
    channel = torch.arange(3, device=base.device).repeat(pixel_indices.numel())
    row = torch.arange(rows, device=base.device)
    design[row, channel] = base_pixels.reshape(-1)
    design[row, 3 + channel] = coverage_pixels[:, None].expand(-1, 3).reshape(-1)
    if normal_conditioned:
        basis_pixels = basis.reshape(3, 3, -1)[:, :, pixel_indices].permute(2, 1, 0)
        design[:, 6:9] = basis_pixels.reshape(-1, 3)
    return design, pixel_indices


def fit_model(
    observed, base, coverage, basis, fit_region, maximum_dimension,
    normal_conditioned,
):
    height, width = coverage.shape
    scale = min(1.0, maximum_dimension / max(height, width))
    size = (max(16, round(height * scale)), max(16, round(width * scale)))
    observed_fit = _resize(observed, size)
    base_fit = _resize(base, size)
    coverage_fit = _resize(coverage, size)[0]
    basis_fit = (torch.stack([_resize(basis[axis], size) for axis in range(3)])
                 if normal_conditioned else None)
    region_fit = _resize(fit_region, size, mode="nearest")[0] > 0.5
    yy, xx = torch.meshgrid(
        torch.arange(size[0], device=base.device),
        torch.arange(size[1], device=base.device),
        indexing="ij",
    )
    checker = ((xx + yy) & 1) == 0
    pixel_mask = region_fit & checker & (coverage_fit > 0.05)
    fallback = False
    if int(pixel_mask.sum().item()) < 64:
        pixel_mask = checker & (coverage_fit > 0.05)
        fallback = True
    if int(pixel_mask.sum().item()) < 64:
        raise RuntimeError("Insufficient covered pixels for illumination fit")

    design, pixel_indices = _design_matrix(
        base_fit, coverage_fit, basis_fit, pixel_mask, normal_conditioned
    )
    target = observed_fit.reshape(3, -1)[:, pixel_indices].T.reshape(-1).float()
    pixel_weight = torch.ones(pixel_indices.numel(), device=base.device)
    identity = torch.tensor(
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        + ([0.0, 0.0, 0.0] if normal_conditioned else []),
        device=base.device,
    )
    coefficient = identity
    for _ in range(4):
        row_weight = pixel_weight[:, None].expand(-1, 3).reshape(-1)
        weighted_design = design * row_weight.sqrt()[:, None]
        weighted_target = target * row_weight.sqrt()
        gram = weighted_design.T @ weighted_design
        ridge = torch.finfo(gram.dtype).eps * torch.trace(gram).clamp_min(1.0)
        coefficient = torch.linalg.solve(
            gram + ridge * torch.eye(gram.shape[0], device=gram.device),
            weighted_design.T @ weighted_target + ridge * identity,
        )
        residual = (design @ coefficient - target).reshape(-1, 3).norm(dim=1)
        scale_estimate = residual.median().clamp_min(1e-4) / 0.6745
        cutoff = 1.345 * scale_estimate
        pixel_weight = torch.where(
            residual <= cutoff, torch.ones_like(residual), cutoff / residual
        )
    return coefficient, fallback, int(pixel_indices.numel())


def predict(base, coverage, basis, coefficient):
    result = coefficient[:3, None, None] * base
    result = result + coefficient[3:6, None, None] * coverage[None]
    if coefficient.numel() == 9:
        result = result + torch.einsum("d,dchw->chw", coefficient[6:9], basis)
    return result.clamp(0.0, 1.0)


def _robust_mse(observed, predicted, region):
    values = (observed - predicted).square().mean(dim=0)[region]
    if values.numel() < 16:
        return float("nan")
    return float(values.clamp_max(torch.quantile(values, 0.90)).mean().item())


def directional_relighting_admissible(
    reference_views, gaussians, pipe, background, normals, object_gaussians,
    fit_max_side,
):
    affine_error = []
    relighting_error = []
    for view in reference_views:
        base, coverage, basis, object_pixels = render_relighting_basis(
            view, gaussians, pipe, background, normals, object_gaussians
        )
        fit_region = (coverage > 0.05) & ~object_pixels
        affine_coefficient, _, _ = fit_model(
            view.original_image[:3], base, coverage, basis, fit_region,
            fit_max_side, False,
        )
        relighting_coefficient, _, _ = fit_model(
            view.original_image[:3], base, coverage, basis, fit_region,
            fit_max_side, True,
        )
        affine = predict(base, coverage, basis, affine_coefficient)
        relighting = predict(base, coverage, basis, relighting_coefficient)
        yy, xx = torch.meshgrid(
            torch.arange(coverage.shape[0], device=coverage.device),
            torch.arange(coverage.shape[1], device=coverage.device),
            indexing="ij",
        )
        heldout = ((xx + yy) & 1) == 1
        heldout_object = heldout & (coverage > 0.05) & object_pixels
        if int(heldout_object.sum().item()) < 16:
            heldout_object = heldout & (coverage > 0.05) & ~object_pixels
        affine_error.append(_robust_mse(view.original_image[:3], affine, heldout_object))
        relighting_error.append(
            _robust_mse(view.original_image[:3], relighting, heldout_object)
        )
    affine_mean = sum(affine_error) / len(affine_error)
    relighting_mean = sum(relighting_error) / len(relighting_error)
    return relighting_mean <= affine_mean, {
        "reference_views": len(reference_views),
        "mean_affine_mse": affine_mean,
        "mean_relighting_mse": relighting_mean,
    }

