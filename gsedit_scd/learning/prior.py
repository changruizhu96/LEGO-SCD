"""Source-image weighting and branch-and-bound fitting of the pixel prior."""
import heapq
import math

import numpy as np


def float32_ceiling(values):
    values = np.asarray(values, dtype=np.float64)
    rounded = values.astype(np.float32)
    return np.where(rounded.astype(np.float64) < values,
                    np.nextafter(rounded, np.float32(np.inf)), rounded)


def tail_integration_cuts():
    """Fixed quadrature nodes in the half-log-odds score coordinate."""
    extent = math.log(0.99999 / 0.00001)
    probabilities = np.unique(np.round(1 / (1 + np.exp(np.linspace(-extent, extent, 257))), 5))[::-1]
    # Keep the original integration locations, including their small offset.
    # These are quadrature nodes, not a restriction on the fitted pixel prior.
    offset = math.log(32768 / 32767) / 2
    return np.array([math.log((1 - value) / value) + offset for value in probabilities])


def image_weights(negative_scores, event_keys, integration_cuts=None):
    cuts = tail_integration_cuts() if integration_cuts is None else np.asarray(integration_cuts, dtype=np.float64)
    search_cuts = float32_ceiling(cuts)
    tails = np.array([(a.size - np.searchsorted(a, search_cuts, side="left")) / a.size
                      for a in negative_scores])
    events = sorted(set(event_keys))
    group = np.array([events.index(key) for key in event_keys])
    event_tails = np.stack([tails[group == g].mean(0) for g in range(len(events))])
    references = np.stack([np.median(np.delete(event_tails, g, axis=0), axis=0)
                           for g in range(len(events))])
    distance = np.trapezoid(np.maximum(tails - references[group], 0), x=cuts, axis=1) / (cuts[-1] - cuts[0])
    event_distances = np.array([np.median(distance[group == g]) for g in range(len(events))])
    centers, scales = [], []
    for g in range(len(events)):
        other = np.delete(event_distances, g)
        center = float(np.median(other))
        scale = float(1.4826 * np.median(abs(other - center)))
        if scale <= 0:
            raise ValueError("Source event tail deviations have zero MAD; the retained weighting is undefined.")
        centers.append(center)
        scales.append(scale)
    z = np.maximum((distance - np.array(centers)[group]) / np.array(scales)[group], 0)
    quality = 1 / (1 + z * z)
    return quality / quality.sum()


def fit_prior(positive_scores, negative_scores, weights, epsilon_pp=0.01):
    """Inputs are sorted per-image half-log-odds scores from source sequences."""
    positives = np.array([len(a) for a in positive_scores])
    if np.any(positives == 0) or any(len(a) == 0 for a in negative_scores):
        raise ValueError("Prior fitting requires changed and unchanged pixels in every scored source image.")
    weights = np.asarray(weights, dtype=np.float64)
    all_scores = positive_scores + negative_scores
    lower = min(float(a[0]) for a in all_scores)
    upper = float(np.nextafter(np.float32(max(float(a[-1]) for a in all_scores)), np.float32(np.inf)))
    heap, best, sequence = [], None, 0

    def evaluate(cut):
        nonlocal best
        cut = float(np.float32(cut))
        tp = np.array([a.size - np.searchsorted(a, np.float32(cut), side="left") for a in positive_scores])
        fp = np.array([a.size - np.searchsorted(a, np.float32(cut), side="left") for a in negative_scores])
        point = (cut, tp, fp, float(weights @ (tp / (positives + fp))))
        if best is None or (point[3], cut) > (best[3], best[0]):
            best = point
        return point

    def enqueue(left, right):
        nonlocal sequence
        bound = float(weights @ (left[1] / (positives + right[2]))) + 1e-12
        if bound <= best[3] or np.nextafter(np.float32(left[0]), np.float32(np.inf)) >= right[0]:
            return
        sequence += 1
        heapq.heappush(heap, (-bound, sequence, left, right))

    left, right = evaluate(lower), evaluate(upper)
    enqueue(left, right)
    while heap and -heap[0][0] - best[3] > epsilon_pp / 100:
        negative_bound, _, left, right = heapq.heappop(heap)
        if -negative_bound <= best[3]:
            continue
        middle = evaluate(float32_ceiling((left[0] + right[0]) / 2).item())
        enqueue(left, middle)
        enqueue(middle, right)
    bound = max(best[3], -heap[0][0]) if heap else best[3]
    excluded, included = [], []
    for values in all_scores:
        index = int(np.searchsorted(values, np.float32(best[0]), side="left"))
        if index:
            excluded.append(float(values[index - 1]))
        if index < values.size:
            included.append(float(values[index]))
    gap_left = max(excluded) if excluded else float(np.nextafter(np.float32(lower), np.float32(-np.inf)))
    gap_right = min(included) if included else upper
    cutoff = (gap_left + gap_right) / 2
    # A floating probability of 0.5 corresponds to zero log odds.
    prior_log_odds = -2.0 * cutoff
    return {"prior_log_odds": prior_log_odds,
            "effective_pixel_prior": 1 / (1 + math.exp(-prior_log_odds)),
            "source_weighted_iou": best[3], "global_upper_bound": bound,
            "optimality_gap_pp": 100 * (bound - best[3]), "epsilon_pp": epsilon_pp,
            "score_gap": [gap_left, gap_right]}
