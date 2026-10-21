"""Shared response correction and image-to-Gaussian evidence pipeline."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .geometry import (
    aggregate_primitive_measurement_from_probe,
    render_primitive_measurement_probe,
    render_probability_map,
)


def mean_filter(image, size):
    padding = size // 2
    horizontal = F.pad(image[None, None], (padding, padding, 0, 0), mode="reflect")
    horizontal = F.avg_pool2d(horizontal, (1, size), stride=1)
    vertical = F.pad(horizontal, (0, 0, padding, padding), mode="reflect")
    return F.avg_pool2d(vertical, (size, 1), stride=1).squeeze(0).squeeze(0)


def gradient_magnitude(image):
    horizontal = torch.zeros_like(image)
    vertical = torch.zeros_like(image)
    horizontal[..., :, :-1] = image[..., :, 1:] - image[..., :, :-1]
    vertical[..., :-1, :] = image[..., 1:, :] - image[..., :-1, :]
    return torch.sqrt(horizontal.square() + vertical.square())


@torch.no_grad()
def suppression_features(response, coverage, reference_rgb, ray_stats):
    """Four response features; the full model adds 13 rendering features."""
    response = response.squeeze().float()
    coverage = coverage.squeeze().float()
    mean7 = mean_filter(response, 7)
    std7 = torch.sqrt((mean_filter(response.square(), 7) - mean7.square()).clamp_min(0.0))
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                           device=response.device, dtype=response.dtype)[None, None]
    padded = F.pad(response[None, None], (1, 1, 1, 1), mode="reflect")
    gx = F.conv2d(padded, sobel_x).squeeze()
    gy = F.conv2d(padded, sobel_x.transpose(-1, -2)).squeeze()
    base = torch.stack((response, mean_filter(response, 21), std7,
                        torch.sqrt(gx.square() + gy.square())), dim=-1)
    alpha_gradient = gradient_magnitude(coverage)
    rgb_gradient = gradient_magnitude(reference_rgb).mean(dim=0)
    ray = torch.stack((ray_stats[1].clamp(0., 1.), ray_stats[2].clamp(0., 1.),
                       ray_stats[3].clamp(0., 1.), torch.log1p(ray_stats[4].clamp_min(0.)),
                       torch.log1p(ray_stats[5].clamp_min(0.)),
                       torch.log1p(ray_stats[6].clamp_min(0.))), dim=-1).float()
    full = torch.cat((base, torch.stack((coverage, alpha_gradient, rgb_gradient), dim=-1), ray,
                      torch.stack((response * alpha_gradient, response * rgb_gradient,
                                   response * ray[..., 2], response * (1. - ray[..., 0])), dim=-1)), dim=-1)
    return base, full


@torch.no_grad()
def suppress_response(response, coverage, reference_rgb, ray_stats, model):
    base, full = suppression_features(response, coverage, reference_rgb, ray_stats)
    logits = []
    for prefix, features in (("base", base), ("full", full)):
        logits.append((((features - model[prefix + "_feature_mean"])
                        / model[prefix + "_feature_scale"] * model[prefix + "_coefficient"])
                       .sum(dim=-1) + float(model[prefix + "_intercept"])))
    correction = (logits[1] - logits[0]).clamp(
        float(model["incremental_clip_lower"]), float(model["incremental_clip_upper"])
    ).clamp_max(0.0)
    correction = torch.where(coverage.squeeze() > 1e-3, correction, 0.0)
    corrected = torch.sigmoid(torch.logit(response.squeeze().float().clamp(1e-5, 1-1e-5)) + correction)
    return torch.where(coverage.squeeze() > 1e-3, corrected, response.squeeze().float())


@torch.no_grad()
def image_evidence(response, log_likelihood_ratio):
    index = torch.floor(response.float().clamp(0.0, 1.0) * log_likelihood_ratio.numel()).long()
    index = index.clamp_max(log_likelihood_ratio.numel() - 1)
    return torch.sigmoid(log_likelihood_ratio[index]).clamp(1e-6, 1-1e-6)


@dataclass
class CurrentEvidence:
    response: torch.Tensor
    image: torch.Tensor
    geometry: torch.Tensor | None
    combined: torch.Tensor
    coverage: torch.Tensor


def current_evidence(view, gaussians, pipe, background, suppression, log_likelihood_ratio,
                     *, use_suppression=True, use_mapping=True, use_geometry=True):
    """The same fitted front end is used for reference fitting and query images."""
    with torch.set_grad_enabled(use_geometry):
        probe, rendered, coverage, rays = render_primitive_measurement_probe(
            view, gaussians, pipe, background,
            return_ray_stats=use_suppression, return_owner_ids=use_suppression
        )
        response = 0.5 * view.probability_measurement
        if use_suppression:
            response = suppress_response(response, coverage, view.reference_render, rays, suppression)
        probability = (image_evidence(response, log_likelihood_ratio) if use_mapping
                       else response.squeeze().float().clamp(1e-6, 1-1e-6))
        if use_geometry:
            primitive_mean, _, _ = aggregate_primitive_measurement_from_probe(probe, rendered, probability)
    support = (render_probability_map(view, gaussians, pipe, background, primitive_mean, coverage)
               if use_geometry else None)
    logits = torch.logit(probability)
    if use_geometry:
        logits = logits + torch.logit(support.float().clamp(1e-6, 1-1e-6))
    combined = torch.sigmoid(logits)
    combined = torch.where(coverage > 0.001, combined, torch.zeros_like(combined))
    return CurrentEvidence(response, probability, support, combined, coverage)
