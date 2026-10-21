"""Unchanged-response moments in the zero-strength Beta limit."""
import torch


@torch.no_grad()
def fit_unchanged_responses(responses, weights, graph):
    responses, weights = responses.float().cpu(), weights.float().cpu()
    global_mean = float((responses * weights).sum().div(weights.sum().clamp_min(1e-8)).clamp(1e-3, 1-1e-3))
    count = weights.sum(dim=0)
    empirical_mean = (weights * responses).sum(dim=0) / count.clamp_min(1e-8)
    empirical_variance = (weights * (responses - empirical_mean).square()).sum(dim=0) / count.clamp_min(1e-8)
    standardized = (responses - empirical_mean) / empirical_variance.clamp_min(1e-6).sqrt()
    source, target, similarity = (value.detach().cpu() for value in graph)
    numerator, denominator = torch.tensor(0.0), torch.tensor(0.0)
    for index in range(responses.shape[0]):
        edge_weight = torch.minimum(weights[index, source], weights[index, target]) * similarity
        numerator += (edge_weight * standardized[index, source] * standardized[index, target]).sum()
        denominator += edge_weight.sum()
    c = float((numerator / denominator).clamp(0.0, 0.95)) if float(denominator) > 0 else 0.0
    positive = (weights * responses).sum(dim=0).double()
    negative = (weights * (1.0 - responses)).sum(dim=0).double()
    total = positive + negative
    observed = total > 0
    mean = torch.where(observed, positive / total.clamp_min(1e-30), global_mean)
    variance = torch.where(observed, mean * (1.0 - mean) / (total + 1.0),
                           global_mean * (1.0 - global_mean))
    return {"mean": mean.float(), "variance": variance.float(),
            "summary": {"global_shrinkage_strength": 0.0, "scene_unchanged_mean": global_mean,
                        "moment_limit": "kappa_to_zero_from_above",
                        "surface_edge_response_correlation": c,
                        "reference_captures": len(responses), "cross_view_count_correction": False}}
