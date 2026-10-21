"""Class-balanced monotone log likelihood-ratio estimation."""
import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression

FINE_BINS = 65536
TABLE_BINS = 4096

def fine_scores():
    return torch.linspace(0.0, 1.0, FINE_BINS, dtype=torch.float64)

def fit_jeffreys_log_lr(h1, hS, hB, N1, NS, NB):
    """Fit monotone blocks with Jeffreys smoothing and constant tails."""
    s_weight = 0.5
    h1 = h1.double() / h1.sum(); hS = hS.double() / hS.sum(); hB = hB.double() / hB.sum()
    h0 = s_weight * hS + (1.0 - s_weight) * hB
    scores = fine_scores()
    pooled = 0.5 * (h0 + h1)
    nz = torch.nonzero(pooled > 0).squeeze(1)
    x = scores[nz].numpy(); w = pooled[nz].numpy()
    y = (h1[nz] / (h0[nz] + h1[nz])).numpy()
    m1 = h1[nz].numpy(); mS = hS[nz].numpy(); mB = hB[nz].numpy()
    fitted = IsotonicRegression(increasing=True, out_of_bounds="clip").fit_transform(x, y, sample_weight=w)
    block = np.concatenate(([0], np.cumsum(np.abs(np.diff(fitted)) > 1e-12)))

    def block_sums(block):
        b = int(block.max()) + 1
        def agg(v): return np.bincount(block, weights=v, minlength=b)
        M1, MS, MB, P = agg(m1), agg(mS), agg(mB), agg(w)
        C = agg(w * x) / np.maximum(P, 1e-300)
        lo = np.full(b, np.inf); np.minimum.at(lo, block, x)
        hi = np.full(b, -np.inf); np.maximum.at(hi, block, x)
        return b, M1, MS, MB, P, C, lo, hi

    merges = 0
    for _ in range(200):
        b, M1, MS, MB, P, C, lo, hi = block_sums(block)
        p1 = (M1 * N1 + 0.5) / (N1 + 0.5 * b)
        pS = (MS * NS + 0.5) / (NS + 0.5 * b)
        pB = (MB * NB + 0.5) / (NB + 0.5 * b)
        L = np.log(p1) - np.log(s_weight * pS + (1.0 - s_weight) * pB)
        if b == 1 or np.all(np.diff(L) >= -1e-12):
            break
        g = IsotonicRegression(increasing=True, out_of_bounds="clip").fit_transform(C, L, sample_weight=P)
        merged = np.concatenate(([0], np.cumsum(np.abs(np.diff(g)) > 1e-12)))
        block = merged[block]; merges += 1
    else:
        raise RuntimeError("Jeffreys block merging did not converge")
    pure_top = bool((MS[-1] + MB[-1]) <= 0.0)
    pure_bottom = bool(M1[0] <= 0.0)
    knots_x, knots_y = C.copy(), L.copy()
    ramp = {"top": False, "bottom": False}
    table_centers = (np.arange(TABLE_BINS, dtype=np.float64) + 0.5) / TABLE_BINS
    table = np.interp(table_centers, knots_x, knots_y)
    return {
        "table": torch.from_numpy(table).double(),
        "edges": None,
        "centers": torch.from_numpy(C).double(),
        "raw_log_lr": torch.from_numpy(L).double(),
        "isotonic_log_lr": torch.from_numpy(L).double(),
        "coarse_h0_mass": torch.from_numpy(s_weight * MS + (1.0 - s_weight) * MB).double(),
        "coarse_h1_mass": torch.from_numpy(M1).double(),
        "tail": "flat",
        "tail_extrapolated": ramp,
        "jeffreys": {
            "N1": int(N1), "NS": int(NS), "NB": int(NB), "blocks": int(b), "merge_passes": int(merges),
            "units": None,
            "pure_top": pure_top, "pure_bottom": pure_bottom,
            "top_log_lr": float(L[-1]), "bottom_log_lr": float(L[0]),
            "top_bound_if_pure": float(np.log((M1[-1] * N1 + 0.5) / (N1 + 0.5 * b)) - np.log(s_weight * 0.5 / (NS + 0.5 * b) + (1.0 - s_weight) * 0.5 / (NB + 0.5 * b))),
            "knots_x": [float(v) for v in knots_x.tolist()],
        },
    }

