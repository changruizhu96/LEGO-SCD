"""Responsibility aggregation, geometric support, and the fixed K8 neighbourhood."""
import numpy as np
import torch
from gsedit_scd.vendor.oscd.gaussian_renderer import render
from gsedit_scd.vendor.oscd.utils.sh_utils import SH2RGB

def render_primitive_measurement_probe(
    view, gaussians, pipe, background, return_ray_stats=False,
    return_owner_ids=False,
):
    probe_colors = torch.full(
        (gaussians.get_xyz.shape[0], 3),
        0.5,
        dtype=gaussians.get_xyz.dtype,
        device=gaussians.get_xyz.device,
        requires_grad=True,
    )
    render_pkg = render(
        view,
        gaussians,
        pipe,
        background,
        override_color=probe_colors,
        return_ray_stats=(return_ray_stats or return_owner_ids),
        return_owner_ids=return_owner_ids,
    )
    rendered_probe = render_pkg["render"]
    coverage = (rendered_probe[1].detach() / 0.5).clamp(0.0, 1.0)
    ray_stats = (
        render_pkg["ray_stats"].detach()
        if return_ray_stats else None
    )
    return probe_colors, rendered_probe, coverage, ray_stats

def aggregate_primitive_measurement_from_probe(
    probe_colors, rendered_probe, measurement, retain_graph=False,
):
    measurement = measurement.squeeze()
    objective = (
        (rendered_probe[0] * measurement).sum()
        + rendered_probe[1].sum()
        + (rendered_probe[2] * measurement.square()).sum()
    )
    responsibility_moments = torch.autograd.grad(
        objective, probe_colors, retain_graph=retain_graph
    )[0]
    support = responsibility_moments[:, 1].clamp_min(0.0)
    primitive_residual = responsibility_moments[:, 0] / support.clamp_min(1e-8)
    primitive_second_moment = responsibility_moments[:, 2] / support.clamp_min(1e-8)
    primitive_variance = (
        primitive_second_moment - primitive_residual.square()
    ).clamp_min(0.0)
    return (
        primitive_residual.detach(),
        primitive_variance.detach(),
        support.detach(),
    )

def aggregate_primitive_measurement(
    view, gaussians, pipe, background, measurement, return_ray_stats=False,
    return_owner_ids=False,
):
    probe_colors, rendered_probe, coverage, ray_stats = (
        render_primitive_measurement_probe(
            view,
            gaussians,
            pipe,
            background,
            return_ray_stats=return_ray_stats,
            return_owner_ids=return_owner_ids,
        )
    )
    result = aggregate_primitive_measurement_from_probe(
        probe_colors,
        rendered_probe,
        measurement,
    ) + (coverage,)
    return result

@torch.no_grad()
def render_probability_map(view, gaussians, pipe, background, posterior, coverage):
    colors = posterior[:, None].expand(-1, 3).contiguous()
    probability_mass = render(
        view, gaussians, pipe, background, override_color=colors
    )["render"].mean(dim=0)
    probability = probability_mass / coverage.clamp_min(1e-6)
    return torch.where(
        coverage > 0.001,
        probability.clamp(0.0, 1.0),
        0.0,
    )

@torch.no_grad()
def build_gaussian_overlap_graph(gaussians, number_of_neighbors, appearance_aware=False):
    from scipy.spatial import cKDTree
    xyz = gaussians.get_xyz.detach()
    neighbor_count = min(number_of_neighbors + 1, xyz.shape[0])
    _, neighbor_index = cKDTree(xyz.cpu().numpy()).query(xyz.cpu().numpy(), k=neighbor_count)
    source = np.repeat(np.arange(xyz.shape[0]), neighbor_count - 1)
    target = neighbor_index[:, 1:].reshape(-1)
    pairs = np.stack((np.minimum(source, target), np.maximum(source, target)), axis=1)
    pairs = np.unique(pairs[pairs[:, 0] != pairs[:, 1]], axis=0)
    source = torch.from_numpy(pairs[:, 0]).long().to(xyz.device)
    target = torch.from_numpy(pairs[:, 1]).long().to(xyz.device)
    compact_covariance = gaussians.get_covariance().detach()
    covariance = torch.zeros((compact_covariance.shape[0], 3, 3), dtype=compact_covariance.dtype, device=compact_covariance.device)
    covariance[:, 0, 0] = compact_covariance[:, 0]
    covariance[:, 0, 1] = covariance[:, 1, 0] = compact_covariance[:, 1]
    covariance[:, 0, 2] = covariance[:, 2, 0] = compact_covariance[:, 2]
    covariance[:, 1, 1] = compact_covariance[:, 3]
    covariance[:, 1, 2] = covariance[:, 2, 1] = compact_covariance[:, 4]
    covariance[:, 2, 2] = compact_covariance[:, 5]
    displacement = xyz[target] - xyz[source]
    pair_covariance = covariance[source] + covariance[target]
    jitter = torch.eye(3, device=xyz.device, dtype=xyz.dtype) * 1e-10
    mahalanobis_squared = (displacement * torch.linalg.solve(pair_covariance + jitter, displacement.unsqueeze(2)).squeeze(2)).sum(dim=1).clamp_min(0.0)
    raw_weight = torch.exp(-0.5 * mahalanobis_squared.clamp_max(50.0))
    color_distance_scale = None
    rgb = SH2RGB(gaussians.get_features.detach()[:, 0]).clamp(0.0, 1.0)
    color_distance = torch.linalg.vector_norm(rgb[target] - rgb[source], dim=1)
    overlapping = raw_weight > 0.0001
    positive_color_distance = color_distance[overlapping & (color_distance > 1e-06)]
    color_distance_scale = torch.median(positive_color_distance).clamp_min(0.001) if positive_color_distance.numel() > 0 else torch.tensor(0.001, device=xyz.device, dtype=xyz.dtype)
    appearance_similarity = torch.exp(-0.5 * (color_distance / color_distance_scale).square())
    raw_weight = raw_weight * appearance_similarity
    retained = raw_weight > 0.0001
    source = source[retained]
    target = target[retained]
    raw_weight = raw_weight[retained]
    weight = raw_weight if appearance_aware else raw_weight / raw_weight.mean().clamp_min(1e-08)
    stats = {'neighbors': number_of_neighbors, 'candidate_edges': int(pairs.shape[0]), 'overlap_edges': int(retained.sum().item()), 'appearance_aware': appearance_aware, 'color_distance_scale': float(color_distance_scale.item()) if color_distance_scale is not None else None, 'raw_weight_quantiles': {str(quantile): float(torch.quantile(raw_weight, quantile).item()) for quantile in (0.05, 0.5, 0.95)}}
    return ((source, target, weight), stats)
