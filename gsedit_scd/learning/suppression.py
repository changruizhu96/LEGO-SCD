"""Fit the response-only and renderer-conditioned logistic models."""
from collections import defaultdict
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

BASE_FEATURE_NAMES = ("raw_input", "raw_mean_21", "raw_std_7", "raw_gradient")
FULL_FEATURE_NAMES = (*BASE_FEATURE_NAMES, "alpha_coverage", "alpha_gradient",
                      "reference_rgb_gradient", "ray_top1_fraction", "ray_top2_fraction",
                      "ray_normalized_entropy", "ray_log_effective_count",
                      "ray_log_relative_depth_std", "ray_log_contributor_count",
                      "raw_times_alpha_gradient", "raw_times_reference_rgb_gradient",
                      "raw_times_ray_normalized_entropy", "raw_times_one_minus_ray_top1_fraction")


def feature_vectors(features, ray, feature_names):
    index = {name: i for i, name in enumerate(feature_names)}
    raw = features[:, index["raw_input"]].float()
    base = torch.stack([features[:, index[name]].float() for name in BASE_FEATURE_NAMES], dim=1)
    alpha = features[:, index["alpha_coverage"]].float()
    alpha_gradient = features[:, index["alpha_gradient"]].float()
    rgb_gradient = features[:, index["reference_rgb_gradient"]].float()
    ray = ray.float()
    full = torch.cat((base, torch.stack((alpha, alpha_gradient, rgb_gradient), dim=-1), ray[..., 1:],
                      torch.stack((raw * alpha_gradient, raw * rgb_gradient,
                                   raw * ray[..., 3], raw * (1.0 - ray[..., 1])), dim=-1)), dim=-1)
    return base, full


def fit_suppression(dataset):
    blocks = load_blocks(dataset)
    train = make_dataset(blocks, "train")
    validation = make_dataset(blocks, "validation")
    # Preserve the fitted model's original train/validation ordering and weights.
    base = np.concatenate((train[0], validation[0]))
    full = np.concatenate((train[1], validation[1]))
    labels = np.concatenate((train[2], validation[2]))
    weights = np.concatenate((train[3], validation[3]))
    base_scaler, base_model = fit_logistic(base, labels, weights)
    full_scaler, full_model = fit_logistic(full, labels, weights)
    correction = (full_model.decision_function(full_scaler.transform(full))
                  - base_model.decision_function(base_scaler.transform(base)))
    return {"model": "response_only_and_renderer_conditioned_logistic",
            "source_scene": dataset["source_scene"],
            "base_feature_names": list(BASE_FEATURE_NAMES), "full_feature_names": list(FULL_FEATURE_NAMES),
            **model_payload("base", base_scaler, base_model),
            **model_payload("full", full_scaler, full_model),
            "incremental_clip_lower": float(np.quantile(correction, 0.01)),
            "incremental_clip_upper": 0.0, "fused_only_features": True,
            "positive_increment_allowed": False, "neutral_h1_protection": False,
            "matched_H0_H1": True, "image_coordinates_used": False,
            "size_features_used": False, "target_views_used": False,
            "target_GT_used": False, "object_score_broadcast": False,
            "source_validation": {"training_samples": len(train[2]), "validation_samples": len(validation[2])}}

def load_blocks(dataset):
    blocks = []
    feature_names = tuple(dataset["feature_names"])
    for record in dataset["matched_records"]:
        regions = record.get("translation_regions")
        if regions is None:
            regions = {record["event"]: record}
        for region_name, pair in regions.items():
            for label, source, family in (
                (1, pair["event_features"], "h1"),
                (0, pair["features"], "h0_matched"),
            ):
                base, full = feature_vectors(
                    source,
                    pair["ray_features"],
                    feature_names,
                )
                blocks.append({
                    "label": label,
                    "object_id": record["object_id"],
                    "row_id": f"{record['row_key']}::{region_name}",
                    "split": record["split"],
                    "family": family,
                    "base": base.numpy().astype(np.float64),
                    "full": full.numpy().astype(np.float64),
                    "pixel_weight": pair["weights"].float().numpy(),
                })
    for record in dataset["hard_records"]:
        base, full = feature_vectors(
            record["features"],
            record["ray_features"],
            feature_names,
        )
        blocks.append({
            "label": 0,
            "object_id": "hard::" + record["group_key"],
            "row_id": "hard::" + record["group_key"],
            "split": record["split"],
            "family": "h0_hard",
            "base": base.numpy().astype(np.float64),
            "full": full.numpy().astype(np.float64),
            "pixel_weight": np.ones(base.shape[0], dtype=np.float64),
        })
    if not blocks:
        raise RuntimeError("frontend nuisance dataset has no blocks")
    return blocks

def make_dataset(blocks, split):
    selected = [block for block in blocks if block["split"] == split]
    family_mass = {"h1": 0.5, "h0_matched": 0.25, "h0_hard": 0.25}
    if {block["family"] for block in selected} != set(family_mass):
        raise RuntimeError(f"frontend nuisance {split} split lacks a family")
    for family, total_mass in family_mass.items():
        family_blocks = [block for block in selected if block["family"] == family]
        by_object = defaultdict(list)
        for block in family_blocks:
            by_object[block["object_id"]].append(block)
        object_mass = total_mass / len(by_object)
        for object_blocks in by_object.values():
            row_mass = object_mass / len(object_blocks)
            for block in object_blocks:
                pixel_weight = block["pixel_weight"].astype(np.float64)
                block["balanced_weight"] = (
                    row_mass * pixel_weight / pixel_weight.sum()
                )
    base = np.concatenate([block["base"] for block in selected])
    full = np.concatenate([block["full"] for block in selected])
    labels = np.concatenate([
        np.full(block["base"].shape[0], block["label"], dtype=np.int64)
        for block in selected
    ])
    weights = np.concatenate([block["balanced_weight"] for block in selected])
    families = np.concatenate([
        np.full(block["base"].shape[0], block["family"], dtype=object)
        for block in selected
    ])
    if not (
        np.isfinite(base).all()
        and np.isfinite(full).all()
        and np.isfinite(weights).all()
        and np.unique(labels).size == 2
    ):
        raise RuntimeError(f"invalid frontend nuisance {split} matrix")
    return base, full, labels, weights, families

def fit_logistic(matrix, labels, weights):
    scaler = StandardScaler().fit(matrix, sample_weight=weights)
    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=0,
    )
    model.fit(scaler.transform(matrix), labels, sample_weight=weights)
    return scaler, model

def model_payload(prefix, scaler, model):
    return {
        f"{prefix}_feature_mean": torch.from_numpy(
            scaler.mean_.astype(np.float32)
        ),
        f"{prefix}_feature_scale": torch.from_numpy(
            scaler.scale_.astype(np.float32)
        ),
        f"{prefix}_coefficient": torch.from_numpy(
            model.coef_[0].astype(np.float32)
        ),
        f"{prefix}_intercept": float(model.intercept_[0]),
    }
