"""Current-method neighbourhood evidence and causal support accumulation."""
import math

import numpy as np
import torch


@torch.no_grad()
def neighbourhood_log_bayes_factor(response, variance, weight, graph, mean,
                                 baseline_variance, correlation):
    """Integrate a unit half-normal enhancement after conditioning on the anchor.

    ``correlation`` is the spatial anchor/neighbour coefficient c. There is no
    cross-view count correction in either this function or SupportField.
    """
    response = response.float().clamp(0.0, 1.0)
    weight = weight.float().clamp(0.0, 1.0)
    total_variance = (baseline_variance.float().clamp_min(0.0)
                      + variance.float().clamp_min(0.0)).clamp_min(1e-6)
    z = ((response - mean.float().clamp(0.0, 1.0))
         / torch.sqrt(total_variance)).clamp(-12.0, 12.0)
    c = torch.as_tensor(correlation, device=response.device, dtype=response.dtype)
    if c.ndim == 0:
        c = c.expand_as(response)

    mass = torch.zeros_like(weight)
    squared_mass = torch.zeros_like(weight)
    first = torch.zeros_like(weight)
    second = torch.zeros_like(weight)
    source, target, similarity = graph
    if source.numel() > 0:
        similarity = similarity.float().clamp(0.0, 1.0)
        source_weight = similarity * weight[target]
        target_weight = similarity * weight[source]
        mass.index_add_(0, source, source_weight)
        mass.index_add_(0, target, target_weight)
        squared_mass.index_add_(0, source, source_weight.square())
        squared_mass.index_add_(0, target, target_weight.square())
        first.index_add_(0, source, source_weight * z[target])
        first.index_add_(0, target, target_weight * z[source])
        second.index_add_(0, source, source_weight * z[target].square())
        second.index_add_(0, target, target_weight * z[source].square())

    observed = mass > 1e-6
    m = torch.where(observed, first / mass.clamp_min(1e-6), 0.0)
    dispersion = torch.where(
        observed, second / mass.clamp_min(1e-6) - m.square(), 0.0
    ).clamp_min(0.0)
    count = torch.where(observed, mass.square() / squared_mass.clamp_min(1e-6), 0.0)
    independent_variance = torch.maximum(torch.ones_like(dispersion), dispersion) / count.clamp_min(1.0)
    neighbourhood_variance = independent_variance * (1.0 + c * (count - 1.0).clamp_min(0.0))
    residual = m - c * z
    residual_variance = (neighbourhood_variance - c.square()).clamp_min(1e-4)
    loading = 1.0 - c
    precision = residual_variance.reciprocal()
    h = precision * loading * residual
    posterior_precision = precision * loading.square() + 1.0
    log_positive_mass = torch.special.log_ndtr(h / torch.sqrt(posterior_precision))
    log_bf = (math.log(2.0) - 0.5 * torch.log(posterior_precision)
              + 0.5 * h.square() / posterior_precision + log_positive_mass)
    return torch.where(observed, log_bf.clamp(-12.0, 12.0), 0.0)


class SupportField:
    """One accumulated log Bayes factor and observation count per Gaussian."""

    def __init__(self, primitive_count, device):
        self.log_evidence = torch.zeros(primitive_count, device=device)
        self.observation_count = torch.zeros_like(self.log_evidence)
        self.frame_count = 0

    @torch.no_grad()
    def read(self):
        return torch.sigmoid(self.log_evidence)

    @torch.no_grad()
    def update(self, log_bayes_factor, weight):
        weight = weight.float().clamp(0.0, 1.0)
        self.log_evidence += log_bayes_factor.float().clamp(-12.0, 12.0) * weight
        self.observation_count += weight
        self.frame_count += 1


def encode_evidence(probability):
    """Encode diagnostic evidence maps as uint16 for PNG export only."""
    return (probability.detach().squeeze().clamp(0.0, 1.0).cpu().numpy() * 65535).astype(np.uint16)


def encode_probability(probability):
    """Encode pixel or refined probabilities as rounded uint16 for PNG export."""
    return np.rint(probability.detach().squeeze().clamp(0.0, 1.0).cpu().numpy() * 65535).astype(np.uint16)


def fusion_score(current, history, covered):
    """Shared floating-point evidence coordinate for fitting and inference."""
    eps = torch.finfo(torch.float32).eps
    current = current.squeeze().float().clamp(eps, 1-eps)
    history = history.squeeze().float().clamp(eps, 1-eps)
    score = 0.5 * (torch.logit(current) + torch.logit(history))
    return torch.where(covered.squeeze(), score, 0.0)


def decode_probability(score, prior_log_odds):
    # Keep the float64 log-odds conversion without intermediate image encoding.
    logits = 2.0 * score.detach().double() + prior_log_odds
    return (1 / (1 + torch.exp(-logits))).float()
