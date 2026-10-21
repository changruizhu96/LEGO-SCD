"""Reference-only sampling for the two response-suppression classifiers."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from gsedit_scd.data.mask_geometry import find_image, make_camera, resized_rgb
from gsedit_scd.online.cues import frame_cues
from gsedit_scd.online.frontend import suppression_features
from gsedit_scd.online.geometry import (
    aggregate_primitive_measurement_from_probe, render_primitive_measurement_probe, render_probability_map,
)
from .suppression import BASE_FEATURE_NAMES

FEATURE_NAMES = (*BASE_FEATURE_NAMES, "alpha_coverage", "alpha_gradient", "reference_rgb_gradient")


def row_key(row):
    return "::".join(row[key] for key in ("object_id", "camera_name", "event", "condition"))


def selected_indices(weights, limit=256):
    indices = torch.nonzero(weights.reshape(-1) > 0, as_tuple=False).squeeze(1)
    if indices.numel() > limit:
        positions = torch.linspace(0, indices.numel() - 1, limit,
                                   dtype=torch.float64, device=indices.device).round().long()
        indices = indices[positions]
    if not indices.numel():
        raise ValueError("A selected edit region has no valid sampling support.")
    return indices


def collect_suppression_samples(replay, rows, learning_cameras):
    gaussians, sam = replay.gaussians, replay.sam
    device = gaussians.get_xyz.device
    pipe, background = replay.pipe, replay.background
    # Split only cameras that contribute samples; empty cameras must not shift
    # the per-split balancing used by the suppression fit.
    sampled_cameras = {row["camera_name"] for row in rows}
    fit_cameras = [name for name in learning_cameras if name in sampled_cameras]
    validation = set(fit_cameras[::5])
    groups = defaultdict(list)
    for row in rows:
        groups[(row["camera_name"], row["condition"])].append(row)
    matched, hard = [], []
    camera_records = json.loads((replay.source / "reference_reconstruction/cameras.json").read_text())
    records = {record["img_name"]: record for record in camera_records}
    cameras = {}
    static_rendering = None

    def edit_rgb(path):
        # Preserve the sample extractor's CPU float32 conversion before upload.
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(image).permute(2, 0, 1).contiguous().to(device)

    def features(observed, reference, view, *, image_features_only=False, image_embeddings=None):
        candidate = frame_cues(observed, reference, sam, view.image_height, view.image_width,
                               normalize_before_resize=True, image_embeddings=image_embeddings)[0]
        response = candidate.squeeze() * 0.5
        nonlocal static_rendering
        if image_features_only:
            coverage, rays = static_rendering
        else:
            with torch.enable_grad():
                probe, rendered, coverage, rays = render_primitive_measurement_probe(
                    view, gaussians, pipe, background, return_ray_stats=True, return_owner_ids=True
                )
                means, _, _ = aggregate_primitive_measurement_from_probe(probe, rendered, response)
            static_rendering = coverage, rays
        base, full = suppression_features(response, coverage, reference, rays)
        # Edited samples use only these features; geometric hard-example
        # selection belongs to the unchanged branch below.
        if image_features_only:
            return full[..., :7]
        raw_support = render_probability_map(view, gaussians, pipe, background, means, coverage)
        ray_features = torch.cat((rays[0].clamp(0, 1)[..., None], full[..., 7:13]), dim=-1)
        return full[..., :7], ray_features, coverage.squeeze(), torch.maximum(response, raw_support.squeeze())

    for (name, condition), entries in sorted(groups.items()):
        # Match the reconstruction camera and saved rendering used to create
        # the edit labels. Suppression samples retain their original LANCZOS resize.
        if name not in cameras:
            with Image.open(entries[0]["reference_image"]) as reference_image:
                max_side = max(reference_image.size)
            image = resized_rgb(find_image(replay.source, name), max_side)
            cameras[name] = make_camera(records[name], image)
        view = cameras[name]
        mask_size = (view.image_width, view.image_height)
        real = view.original_image[:3]
        identity = edit_rgb(entries[0]["reference_image"])
        reference = identity
        unchanged = edit_rgb(entries[0]["h0_image"])
        base_features, rays, coverage, hard_score = features(
            (real + unchanged - identity).clamp(0, 1), reference, view
        )
        ignore = torch.zeros_like(coverage, dtype=torch.bool)
        for row in entries:
            if row.get("h0_ignore_mask"):
                ignore |= replay.mask(row["h0_ignore_mask"], size=mask_size)
        valid_indices = torch.nonzero(((coverage > 0.001) & ~ignore).flatten(), as_tuple=False).squeeze(1)
        order = torch.argsort(hard_score.flatten()[valid_indices], descending=True, stable=True)
        indices = valid_indices[order[:1024]]
        split = "validation" if name in validation else "train"
        flat_base, flat_ray = base_features.reshape(-1, 7), rays.reshape(-1, 7)
        hard.append({"group_key": f"{name}::{condition}", "split": split,
                     "features": flat_base[indices].half().cpu(),
                     "ray_features": flat_ray[indices].half().cpu(),
                     "flat_indices": indices.int().cpu()})
        reference_embedding = None
        for row_index, row in enumerate(entries):
            target = replay.mask(row["target_map"], soft=True, size=mask_size).clamp(0, 1)
            pair_ignore = replay.mask(row["ignore_mask"], size=mask_size)
            if row.get("h0_ignore_mask"):
                pair_ignore |= replay.mask(row["h0_ignore_mask"], size=mask_size)
            valid = (coverage > 0.001) & ~pair_ignore
            if row_index % 2 == 0:
                observed_batch = [(real + edit_rgb(entry["event_image"]) - identity).clamp(0, 1)
                                  for entry in entries[row_index:row_index + 2]]
                images = [F.interpolate(image[None], size=(1024, 1024), mode="nearest")[0]
                          for image in observed_batch]
                with torch.inference_mode():
                    if reference_embedding is None:
                        ref = F.interpolate(reference[None], size=(1024, 1024), mode="nearest")[0]
                        reference_embedding = sam.get_image_embeddings(torch.stack((ref, ref)).half())[-1][0].clone()
                    query_embeddings = sam.get_image_embeddings(
                        torch.stack(images if len(images) == 2 else images * 2).half()
                    )[-1]
            index = row_index % 2
            pair = torch.stack((reference_embedding, query_embeddings[index]))
            event_features = features(observed_batch[index], reference, view,
                                      image_features_only=True, image_embeddings=pair).reshape(-1, 7)
            regions = {"event": target * valid}
            if row["event"] == "translated":
                path = Path(row["reference_image"]).with_name(f"{name}_removal_target_map.png")
                old = replay.mask(path, soft=True, size=mask_size).clamp(0, 1)
                inserted = (target - old).clamp_min(0)
                new = torch.where(old < 1 - 1e-4, inserted / (1 - old).clamp_min(1e-4), 0).clamp(0, 1)
                regions = {"removed_from": (old * (1 - new)).clamp(0, 1) * valid,
                           "inserted_to": inserted * valid}
            region_records = {}
            for region, weight in regions.items():
                indices = selected_indices(weight)
                region_records[region] = {
                    "features": flat_base[indices].half().cpu(),
                    "event_features": event_features[indices].half().cpu(),
                    "ray_features": flat_ray[indices].half().cpu(),
                    "weights": weight.flatten()[indices].half().cpu(),
                    "flat_indices": indices.int().cpu(),
                }
            record = {**{key: row[key] for key in ("object_id", "event", "camera_name", "condition")},
                      "row_key": row_key(row), "split": split}
            if set(region_records) == {"event"}:
                record.update(region_records["event"])
            else:
                record["translation_regions"] = region_records
            matched.append(record)
        print(f"Suppression samples: {name} {condition}", flush=True)
    return {"source_scene": replay.source.name, "feature_names": list(FEATURE_NAMES),
            "matched_records": matched, "hard_records": hard}
