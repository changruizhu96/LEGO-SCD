from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
import cv2
import numpy as np
from scipy.stats import hypergeom
import torch
import torch.nn.functional as functional
from transformers import pipeline
from transformers.pipelines import MaskGenerationPipeline
from gsedit_scd.vendor.oscd.gaussian_renderer import render
from gsedit_scd.vendor.oscd.scene.gaussian_model import GaussianModel
from gsedit_scd.vendor.oscd.utils.general_utils import build_rotation
from gsedit_scd.data.mask_geometry import labeled_panel, lift_mask, make_camera, overlay, owner_observation, project_membership, resized_reference_images, select_contiguous_views

class EarlyQualityMaskPipeline(MaskGenerationPipeline):
    """Apply the same IoU rejection before expanding masks to image resolution."""

    def _ensure_tensor_on_device(self, inputs, device):
        if isinstance(inputs, dict) and "masks" in inputs:
            # filter_masks returns RLE lists of Python integers, not tensors.
            return {name: value if name == "masks" else super(
                EarlyQualityMaskPipeline, self
            )._ensure_tensor_on_device(value, device) for name, value in inputs.items()}
        return super()._ensure_tensor_on_device(inputs, device)

    def _forward(self, model_inputs, pred_iou_thresh=0.88, stability_score_thresh=0.95,
                 mask_threshold=0, stability_score_offset=1):
        input_boxes = model_inputs.pop("input_boxes")
        is_last = model_inputs.pop("is_last")
        original_sizes = model_inputs.pop("original_sizes").tolist()
        model_inputs.pop("reshaped_input_sizes")
        output = self.model(**model_inputs)
        scores = output["iou_scores"][0].flatten()
        masks = output["pred_masks"][0].flatten(0, 1)
        keep = ((scores > pred_iou_thresh).nonzero().flatten() if pred_iou_thresh > 0
                else torch.arange(len(scores), device=scores.device))
        # Interpolation requires a nonempty channel dimension. If all candidates
        # fail, pass one through; the unchanged filter below rejects it.
        if keep.numel() == 0:
            keep = torch.zeros(1, dtype=torch.long, device=scores.device)
        masks = self.image_processor.post_process_masks(
            masks[keep][None, None], original_sizes,
            mask_threshold=mask_threshold, binarize=False,
        )
        masks, scores, boxes = self.image_processor.filter_masks(
            masks[0], scores[keep][None], original_sizes[0], input_boxes[0],
            pred_iou_thresh, stability_score_thresh, mask_threshold, stability_score_offset,
        )
        return {"masks": masks, "is_last": is_last, "boxes": boxes, "iou_scores": scores}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build multi-view latent object hypotheses directly on a frozen "
            "FastGS reconstruction."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views", type=int, default=12)
    parser.add_argument("--neighbor-window", type=int, default=2)
    parser.add_argument("--max-side", type=int, default=1008)
    parser.add_argument(
        "--resolution-divisor", type=int, default=0,
        help=("Match O-SCD by independently rounding the source width and "
              "height divided by this value"),
    )
    parser.add_argument("--points-per-batch", type=int, default=64)
    parser.add_argument("--fdr", type=float, default=0.01)
    parser.add_argument("--visualizations", type=int, default=8)
    parser.add_argument("--sam-model", default="facebook/sam2.1-hiera-tiny")
    parser.add_argument(
        "--observation-cache",
        type=Path,
        help="Optional reusable cache for reference-view SAM masks.",
    )
    parser.add_argument(
        "--allow-relocated-observation-cache",
        action="store_true",
        help=(
            "Allow only the repository-root prefix of the cached source path "
            "to change; instance, scene, cameras, resolution, and SAM settings "
            "remain strictly checked."
        ),
    )
    return parser.parse_args()


def automatic_masks(generator, image, points_per_batch):
    output = generator(image, points_per_batch=points_per_batch)
    image_area = image.width * image.height
    records = []
    for sam_index, (mask, score) in enumerate(zip(output["masks"], output["scores"])):
        mask = mask.bool().cpu()
        area = int(mask.sum().item())
        if area == 0:
            continue
        records.append({
            "sam_index": sam_index,
            "score": float(score),
            "area_fraction": area / image_area,
            "mask": mask,
        })
    return records


def pack_masks(masks):
    packed = []
    for record in masks:
        mask = record["mask"].numpy()
        packed.append({
            "sam_index": record["sam_index"],
            "score": record["score"],
            "area_fraction": record["area_fraction"],
            "shape": list(mask.shape),
            "packed_mask": torch.from_numpy(np.packbits(mask.reshape(-1))),
        })
    return packed


def unpack_masks(records):
    masks = []
    for record in records:
        shape = tuple(record["shape"])
        packed = record["packed_mask"].cpu().numpy()
        mask = np.unpackbits(packed, count=int(np.prod(shape))).reshape(shape)
        masks.append({
            "sam_index": int(record["sam_index"]),
            "score": float(record["score"]),
            "area_fraction": float(record["area_fraction"]),
            "mask": torch.from_numpy(mask.astype(bool)),
        })
    return masks


def same_scene_location(first, second):
    first = Path(first)
    second = Path(second)
    return (
        first.name.casefold(),
        first.parent.name.casefold(),
    ) == (
        second.name.casefold(),
        second.parent.name.casefold(),
    )


def load_observation_cache(path, args, selected_indices, camera_records):
    if path is None or not path.exists():
        return None
    cache = torch.load(path, map_location="cpu", weights_only=False)
    expected_names = [camera_records[index]["img_name"] for index in selected_indices]
    checks = {
        "source": str(args.source),
        "selected_camera_indices": selected_indices,
        "selected_camera_names": expected_names,
        "max_side": args.max_side,
        "resolution_divisor": args.resolution_divisor,
        "sam_model": args.sam_model,
        "points_per_batch": args.points_per_batch,
    }
    for key, expected in checks.items():
        actual = (
            cache.get(key, 0)
            if key == "resolution_divisor"
            else cache.get(key)
        )
        relocated_source = (
            key == "source"
            and args.allow_relocated_observation_cache
            and actual is not None
            and same_scene_location(actual, expected)
        )
        if actual != expected and not relocated_source:
            raise ValueError(
                f"Observation cache {key} mismatch: {actual!r} != {expected!r}"
            )
    return cache


class DisjointViewUnionFind:
    def __init__(self, fragments):
        self.parent = list(range(len(fragments)))
        self.fragments = [[index] for index in range(len(fragments))]
        self.views = [{fragment["view_index"]} for fragment in fragments]

    def find(self, index):
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def merge(self, first, second):
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return False
        if self.views[first_root] & self.views[second_root]:
            return False
        if len(self.fragments[first_root]) < len(self.fragments[second_root]):
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        self.fragments[first_root].extend(self.fragments[second_root])
        self.views[first_root].update(self.views[second_root])
        return True

    def components(self):
        return [
            self.fragments[index]
            for index in range(len(self.parent))
            if self.find(index) == index
        ]


def overlap_edge(first, second, universe_size):
    first_ids = first["gaussian_ids"]
    second_ids = second["gaussian_ids"]
    intersection = np.intersect1d(
        first_ids, second_ids, assume_unique=True
    ).size
    if intersection == 0:
        return None
    union = first_ids.size + second_ids.size - intersection
    p_value = float(hypergeom.sf(
        intersection - 1,
        universe_size,
        first_ids.size,
        second_ids.size,
    ))
    return {
        "first": first["fragment_id"],
        "second": second["fragment_id"],
        "first_view": first["view_index"],
        "second_view": second["view_index"],
        "intersection": int(intersection),
        "jaccard": float(intersection / union),
        "first_containment": float(intersection / first_ids.size),
        "second_containment": float(intersection / second_ids.size),
        "p_value": p_value,
    }


def benjamini_hochberg(edges, false_discovery_rate):
    if not edges:
        return set(), 0.0
    order = sorted(range(len(edges)), key=lambda index: edges[index]["p_value"])
    cutoff_rank = 0
    cutoff = 0.0
    count = len(edges)
    for rank, edge_index in enumerate(order, start=1):
        threshold = false_discovery_rate * rank / count
        if edges[edge_index]["p_value"] <= threshold:
            cutoff_rank = rank
            cutoff = edges[edge_index]["p_value"]
    return set(order[:cutoff_rank]), cutoff


def mutual_best_edges(edges, significant_indices):
    by_pair = {}
    for edge_index in significant_indices:
        edge = edges[edge_index]
        pair = (edge["first_view"], edge["second_view"])
        by_pair.setdefault(pair, []).append(edge_index)
    accepted = []
    for pair_edges in by_pair.values():
        best_first = {}
        best_second = {}
        for edge_index in pair_edges:
            edge = edges[edge_index]
            score = (edge["jaccard"], edge["intersection"])
            first = edge["first"]
            second = edge["second"]
            if first not in best_first or score > best_first[first][0]:
                best_first[first] = (score, edge_index)
            if second not in best_second or score > best_second[second][0]:
                best_second[second] = (score, edge_index)
        for edge_index in pair_edges:
            edge = edges[edge_index]
            if (
                best_first[edge["first"]][1] == edge_index
                and best_second[edge["second"]][1] == edge_index
            ):
                accepted.append(edge_index)
    return accepted


def component_union_ids(component, fragments, exclude_fragment=None):
    members = [
        fragments[index]["gaussian_ids"]
        for index in component
        if index != exclude_fragment
    ]
    if not members:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(members))


@torch.no_grad()
def project_ids(gaussian_ids, view, num_gaussians, device):
    membership = torch.zeros(num_gaussians, device=device)
    if gaussian_ids.size:
        membership[torch.from_numpy(gaussian_ids).to(device)] = 1.0
    probability = project_membership(
        membership,
        view["owner_ids"].to(device),
        view["owner_weights"].to(device),
    )
    return probability.cpu()


def mask_iou(first, second):
    intersection = torch.logical_and(first, second).sum().item()
    union = torch.logical_or(first, second).sum().item()
    return intersection / union if union else 0.0


@torch.no_grad()
def render_geometry_maps(camera, gaussians, pipe, gaussian_normals):
    points = gaussians.get_xyz
    homogeneous = torch.cat((
        points,
        torch.ones((points.shape[0], 1), device=points.device, dtype=points.dtype),
    ), dim=1)
    camera_depth = (homogeneous @ camera.world_view_transform)[:, 2]
    valid_depth = camera_depth > camera.znear
    inverse_depth = torch.zeros_like(camera_depth)
    inverse_depth[valid_depth] = camera_depth[valid_depth].reciprocal()
    inverse_depth_scale = torch.quantile(
        inverse_depth[valid_depth], 0.99
    ).clamp_min(1e-6)
    normalized_inverse_depth = (
        inverse_depth / inverse_depth_scale
    ).clamp(0.0, 1.0)
    depth_probe = torch.stack((
        normalized_inverse_depth,
        normalized_inverse_depth.square(),
        torch.ones_like(normalized_inverse_depth),
    ), dim=1)
    black = torch.zeros(3, device=points.device, dtype=points.dtype)
    depth_render = render(
        camera, gaussians, pipe, black, override_color=depth_probe
    )["render"]
    coverage = depth_render[2].clamp(0.0, 1.0)
    normalized_mean = depth_render[0] / coverage.clamp_min(1e-6)
    rendered_inverse_depth = normalized_mean * inverse_depth_scale

    view_direction = camera.camera_center.reshape(1, 3) - points
    orientation = torch.where(
        (gaussian_normals * view_direction).sum(dim=1, keepdim=True) >= 0.0,
        1.0,
        -1.0,
    )
    oriented_normals = gaussian_normals * orientation
    encoded_normals = 0.5 * (oriented_normals + 1.0)
    normal_render = render(
        camera, gaussians, pipe, black, override_color=encoded_normals
    )["render"]
    rendered_normals = 2.0 * (
        normal_render / coverage.clamp_min(1e-6).unsqueeze(0)
    ) - 1.0
    rendered_normals = functional.normalize(rendered_normals, dim=0, eps=1e-6)
    observable = coverage > 1e-3
    return (
        torch.where(observable, rendered_inverse_depth, 0.0).cpu().numpy(),
        rendered_normals.permute(1, 2, 0).cpu().numpy(),
        observable.cpu().numpy(),
    )


def pairwise_edge_features(image, inverse_depth, normals, observable):
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    log_inverse_depth = np.log(np.maximum(inverse_depth, 1e-8))

    def features(first, second, valid):
        rgb_delta = np.mean(np.abs(rgb[first] - rgb[second]), axis=-1)
        depth_delta = np.abs(log_inverse_depth[first] - log_inverse_depth[second])
        normal_delta = 1.0 - np.clip(np.abs(
            np.sum(normals[first] * normals[second], axis=-1)
        ), 0.0, 1.0)
        return {
            "rgb": (rgb_delta, np.ones_like(valid, dtype=bool)),
            "inverse_depth": (depth_delta, valid),
            "normal": (normal_delta, valid),
        }

    return {
        "horizontal": features(
            (slice(None), slice(None, -1)),
            (slice(None), slice(1, None)),
            observable[:, :-1] & observable[:, 1:],
        ),
        "vertical": features(
            (slice(None, -1), slice(None)),
            (slice(1, None), slice(None)),
            observable[:-1, :] & observable[1:, :],
        ),
    }


def empirical_separation_auc(positive, control):
    if positive.size == 0 or control.size == 0:
        return None
    sorted_control = np.sort(control)
    strictly_lower = np.searchsorted(sorted_control, positive, side="left")
    lower_or_equal = np.searchsorted(sorted_control, positive, side="right")
    return float(np.mean(
        strictly_lower + 0.5 * (lower_or_equal - strictly_lower)
    ) / sorted_control.size)


def mask_boundary_evidence(mask, edge_features):
    mask = np.asarray(mask, dtype=bool)
    mask_u8 = mask.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    local_band = cv2.dilate(mask_u8, kernel, iterations=1).astype(bool)
    local_band ^= cv2.erode(mask_u8, kernel, iterations=1).astype(bool)
    pair_specs = (
        (
            "horizontal",
            mask[:, :-1],
            mask[:, 1:],
            local_band[:, :-1] | local_band[:, 1:],
        ),
        (
            "vertical",
            mask[:-1, :],
            mask[1:, :],
            local_band[:-1, :] | local_band[1:, :],
        ),
    )
    result = {}
    for metric in ("rgb", "inverse_depth", "normal"):
        positive_parts = []
        control_parts = []
        for orientation, first_mask, second_mask, near_boundary in pair_specs:
            values, valid = edge_features[orientation][metric]
            crossing = (first_mask != second_mask) & valid
            local_control = (first_mask == second_mask) & near_boundary & valid
            positive_parts.append(values[crossing])
            control_parts.append(values[local_control])
        positive = np.concatenate(positive_parts)
        control = np.concatenate(control_parts)
        result[f"{metric}_boundary_auc"] = empirical_separation_auc(positive, control)
        result[f"{metric}_boundary_crossing_count"] = int(positive.size)
        result[f"{metric}_boundary_control_count"] = int(control.size)
    return result


def aggregate_boundary_evidence(component, fragments):
    result = {}
    for metric in ("rgb", "inverse_depth", "normal"):
        auc_key = f"{metric}_boundary_auc"
        count_key = f"{metric}_boundary_crossing_count"
        valid = [
            (fragments[index][auc_key], fragments[index][count_key])
            for index in component
            if fragments[index][auc_key] is not None
            and fragments[index][count_key] > 0
        ]
        count = sum(item[1] for item in valid)
        result[auc_key] = (
            float(sum(value * weight for value, weight in valid) / count)
            if count > 0
            else None
        )
        result[count_key] = count
    return result


def component_geometry(
    component,
    fragments,
    gaussian_ids,
    gaussian_xyz,
    gaussian_normals,
    gaussian_flatness,
):
    points = gaussian_xyz[gaussian_ids]
    if points.shape[0] < 3:
        eigenvalues = np.zeros(3, dtype=np.float64)
    else:
        eigenvalues = np.linalg.eigvalsh(np.cov(points.T)).clip(min=0.0)
    largest = max(float(eigenvalues[-1]), 1e-12)
    repeated_ids = np.concatenate([
        fragments[index]["gaussian_ids"] for index in component
    ])
    support_ids, support_counts = np.unique(repeated_ids, return_counts=True)
    core = support_counts * 2 >= len(component)
    core_ids = support_ids[core]
    core_counts = support_counts[core].astype(np.float64)
    if core_ids.size < 3:
        core_ids = support_ids
        core_counts = support_counts.astype(np.float64)
    normals = gaussian_normals[core_ids]
    normal_second_moment = np.einsum(
        "n,ni,nj->ij", core_counts, normals, normals
    ) / core_counts.sum()
    normal_eigenvalues = np.linalg.eigvalsh(normal_second_moment).clip(min=0.0)
    normal_probabilities = normal_eigenvalues / max(normal_eigenvalues.sum(), 1e-12)
    normal_entropy = -float(np.sum(
        np.where(
            normal_probabilities > 0.0,
            normal_probabilities * np.log(normal_probabilities.clip(min=1e-12)),
            0.0,
        )
    ) / np.log(3.0))
    return {
        "centroid": points.mean(axis=0).tolist(),
        "axis_standard_deviations": np.sqrt(eigenvalues[::-1]).tolist(),
        "linearity": float((eigenvalues[-1] - eigenvalues[-2]) / largest),
        "planarity": float((eigenvalues[-2] - eigenvalues[0]) / largest),
        "scattering": float(eigenvalues[0] / largest),
        "normal_diversity": float(1.0 - normal_probabilities[-1]),
        "normal_entropy": normal_entropy,
        "core_gaussian_count": int(core_ids.size),
        "core_gaussian_fraction": float(core_ids.size / gaussian_ids.size),
        "median_primitive_flatness": float(np.median(gaussian_flatness[core_ids])),
    }


def evaluate_components(
    components,
    fragments,
    views,
    num_gaussians,
    device,
    gaussian_xyz,
    gaussian_normals,
    gaussian_flatness,
):
    records = []
    for component_id, component in enumerate(components):
        union_ids = component_union_ids(component, fragments)
        loo_records = []
        for fragment_index in component:
            fragment = fragments[fragment_index]
            without_ids = component_union_ids(
                component, fragments, exclude_fragment=fragment_index
            )
            if without_ids.size == 0:
                continue
            view = views[fragment["view_index"]]
            projection = project_ids(without_ids, view, num_gaussians, device)
            predicted = projection >= 0.5
            target = view["masks"][fragment["mask_index"]]["mask"]
            loo_records.append({
                "fragment_id": fragment_index,
                "view_index": fragment["view_index"],
                "iou": mask_iou(predicted, target),
                "projected_area_fraction": float(predicted.float().mean().item()),
            })
        loo_ious = [record["iou"] for record in loo_records]
        fragment_areas = [fragments[index]["area_fraction"] for index in component]
        records.append({
            "component_id": component_id,
            "fragment_ids": component,
            "view_indices": sorted({fragments[index]["view_index"] for index in component}),
            "view_count": len(component),
            "gaussian_ids": union_ids,
            "gaussian_count": int(union_ids.size),
            "mean_loo_iou": float(np.mean(loo_ious)) if loo_ious else None,
            "median_loo_iou": float(np.median(loo_ious)) if loo_ious else None,
            "mean_fragment_area_fraction": float(np.mean(fragment_areas)),
            "minimum_fragment_area_fraction": float(np.min(fragment_areas)),
            "maximum_fragment_area_fraction": float(np.max(fragment_areas)),
            **aggregate_boundary_evidence(component, fragments),
            **component_geometry(
                component,
                fragments,
                union_ids,
                gaussian_xyz,
                gaussian_normals,
                gaussian_flatness,
            ),
            "loo_records": loo_records,
        })
    return records


def hierarchy_candidates(components, num_gaussians):
    candidates = []
    for child_index, child in enumerate(components):
        child_ids = child["gaussian_ids"]
        best = None
        for parent_index, parent in enumerate(components):
            if child_index == parent_index:
                continue
            parent_ids = parent["gaussian_ids"]
            if parent_ids.size <= child_ids.size:
                continue
            intersection = np.intersect1d(
                child_ids, parent_ids, assume_unique=True
            ).size
            if intersection == 0:
                continue
            candidate = {
                "child_component": child["component_id"],
                "parent_component": parent["component_id"],
                "child_containment": float(intersection / child_ids.size),
                "jaccard": float(
                    intersection / (child_ids.size + parent_ids.size - intersection)
                ),
                "p_value": float(hypergeom.sf(
                    intersection - 1,
                    num_gaussians,
                    child_ids.size,
                    parent_ids.size,
                )),
            }
            score = (candidate["child_containment"], candidate["jaccard"])
            if best is None or score > best[0]:
                best = (score, candidate)
        if best is not None:
            candidates.append(best[1])
    return candidates


def write_component_visualization(
    output,
    components,
    fragments,
    views,
    count,
    num_gaussians,
    device,
    order,
    filename,
):
    def format_auc(value):
        return "n/a" if value is None else f"{value:.2f}"

    candidates = [component for component in components if component["view_count"] >= 2]
    if order == "volumetric":
        candidates.sort(
            key=lambda item: (
                item["scattering"],
                item["mean_loo_iou"] if item["mean_loo_iou"] is not None else -1.0,
                item["view_count"],
            ),
            reverse=True,
        )
    elif order == "normal_diversity":
        candidates.sort(
            key=lambda item: (
                item["normal_diversity"],
                item["mean_loo_iou"] if item["mean_loo_iou"] is not None else -1.0,
                item["view_count"],
            ),
            reverse=True,
        )
    elif order in ("inverse_depth_boundary", "appearance_boundary"):
        metric = (
            "inverse_depth_boundary_auc"
            if order == "inverse_depth_boundary"
            else "rgb_boundary_auc"
        )
        candidates = [item for item in candidates if item[metric] is not None]
        candidates.sort(
            key=lambda item: (
                item[metric],
                item["mean_loo_iou"] if item["mean_loo_iou"] is not None else -1.0,
                item["view_count"],
            ),
            reverse=True,
        )
    else:
        candidates.sort(
            key=lambda item: (
                item["view_count"],
                item["mean_loo_iou"] if item["mean_loo_iou"] is not None else -1.0,
            ),
            reverse=True,
        )
    rows = []
    for component in candidates[:count]:
        loo = max(component["loo_records"], key=lambda item: item["iou"])
        fragment = fragments[loo["fragment_id"]]
        view = views[fragment["view_index"]]
        target_mask = view["masks"][fragment["mask_index"]]["mask"]
        without_ids = component_union_ids(
            component["fragment_ids"], fragments, exclude_fragment=fragment["fragment_id"]
        )
        prediction = project_ids(without_ids, view, num_gaussians, device) >= 0.5
        source_panel = labeled_panel(
            overlay(view["image"], target_mask, (255, 64, 64)),
            f"held mask in {view['name']}",
        )
        prediction_panel = labeled_panel(
            overlay(view["image"], prediction, (64, 220, 255)),
            f"LOO projection IoU {loo['iou']:.3f}",
        )
        union_projection = project_ids(
            component["gaussian_ids"], view, num_gaussians, device
        ) >= 0.5
        union_panel = labeled_panel(
            overlay(view["image"], union_projection, (80, 255, 80)),
            (
                f"object {component['component_id']}: {component['view_count']} views, "
                f"RGB/depth/normal AUC "
                f"{format_auc(component['rgb_boundary_auc'])}/"
                f"{format_auc(component['inverse_depth_boundary_auc'])}/"
                f"{format_auc(component['normal_boundary_auc'])}"
            ),
        )
        rows.append(np.concatenate((source_panel, prediction_panel, union_panel), axis=1))
    if rows:
        cv2.imwrite(
            str(output / filename),
            cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR),
        )


@torch.no_grad()
def write_object_removal(output, components, fragments, views, gaussians, pipe, background):
    candidates = [
        component
        for component in components
        if component["view_count"] >= 3
        and component["mean_loo_iou"] is not None
        and component["inverse_depth_boundary_auc"] is not None
        and component["gaussian_count"] > 0
    ]
    if not candidates:
        return None
    selected = max(
        candidates,
        key=lambda item: (
            item["inverse_depth_boundary_auc"],
            item["mean_loo_iou"],
            item["view_count"],
        ),
    )
    fragment = fragments[selected["fragment_ids"][0]]
    view = views[fragment["view_index"]]
    opacity = gaussians.get_opacity.detach().clone()
    ids = torch.from_numpy(selected["gaussian_ids"]).to(opacity.device)
    opacity[ids] = 0.0
    default_render = render(view["camera"], gaussians, pipe, background)["render"]
    removed_render = render(
        view["camera"], gaussians, pipe, background, override_opacity=opacity
    )["render"]

    def tensor_image(tensor):
        return (
            tensor.clamp(0.0, 1.0)
            .mul(255.0)
            .byte()
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

    projected = project_ids(
        selected["gaussian_ids"],
        view,
        gaussians.get_xyz.shape[0],
        opacity.device,
    ) >= 0.5
    observed_panel = labeled_panel(
        overlay(view["image"], projected, (64, 220, 255)),
        f"latent object {selected['component_id']}",
    )
    render_panel = labeled_panel(
        overlay(tensor_image(default_render), projected, (80, 255, 80)),
        (
            f"{selected['view_count']} views, LOO {selected['mean_loo_iou']:.3f}, "
            f"depth boundary AUC {selected['inverse_depth_boundary_auc']:.3f}"
        ),
    )
    removal_panel = labeled_panel(
        tensor_image(removed_render),
        f"remove {selected['gaussian_count']} Gaussians",
    )
    cv2.imwrite(
        str(output / "latent_object_removal.png"),
        cv2.cvtColor(
            np.concatenate((observed_panel, render_panel, removal_panel), axis=1),
            cv2.COLOR_RGB2BGR,
        ),
    )
    return selected["component_id"]


def serializable_component(component):
    return {
        key: value
        for key, value in component.items()
        if key != "gaussian_ids"
    }


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.views < 2 or args.neighbor_window < 1:
        raise ValueError("views must be >=2 and neighbor-window must be positive")
    if not 0.0 < args.fdr < 1.0:
        raise ValueError("--fdr must lie in (0, 1)")
    if args.resolution_divisor < 0:
        raise ValueError("--resolution-divisor must be zero or positive")
    args.output.mkdir(parents=True)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.benchmark = True
    started = time.perf_counter()

    camera_records = json.loads(
        (args.source / "reference_reconstruction" / "cameras.json").read_text()
    )
    selected_indices = select_contiguous_views(camera_records, args.views)
    observation_cache = load_observation_cache(
        args.observation_cache, args, selected_indices, camera_records
    )
    model_path = (
        args.source
        / "reference_reconstruction"
        / "point_cloud"
        / "iteration_30000"
        / "point_cloud.ply"
    )
    gaussians = GaussianModel(3, 3)
    gaussians.load_ply(str(model_path))
    num_gaussians = gaussians.get_xyz.shape[0]
    device = gaussians.get_xyz.device
    pipe = SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
    )
    background = torch.zeros(3, device=device)
    gaussian_scaling_tensor = gaussians.get_scaling.detach()
    gaussian_rotation_tensor = build_rotation(gaussians.get_rotation.detach())
    smallest_axis = gaussian_scaling_tensor.argmin(dim=1)
    gaussian_normals_tensor = gaussian_rotation_tensor[
        torch.arange(num_gaussians, device=device), :, smallest_axis
    ]
    generator = None
    if observation_cache is None:
        generator = pipeline(
            "mask-generation",
            pipeline_class=EarlyQualityMaskPipeline,
            model=args.sam_model,
            device=0,
            dtype="float32",
            model_kwargs={"local_files_only": True},
        )

    views = []
    fragments = []
    cached_views = []
    reference_images = resized_reference_images(
        args.source, [camera_records[i]["img_name"] for i in selected_indices],
        args.max_side, args.resolution_divisor,
    )
    for local_view_index, camera_index in enumerate(selected_indices):
        record = camera_records[camera_index]
        image = reference_images.pop(record["img_name"])
        camera = make_camera(record, image)
        if observation_cache is None:
            masks = automatic_masks(generator, image, args.points_per_batch)
            cached_views.append({
                "camera_index": camera_index,
                "name": record["img_name"],
                "masks": pack_masks(masks),
            })
        else:
            cached_view = observation_cache["views"][local_view_index]
            if (
                cached_view["camera_index"] != camera_index
                or cached_view["name"] != record["img_name"]
            ):
                raise ValueError("Observation cache view ordering mismatch")
            masks = unpack_masks(cached_view["masks"])
        owner_ids, owner_weights, coverage, _ = owner_observation(
            camera, gaussians, pipe, background
        )
        inverse_depth, rendered_normals, observable = render_geometry_maps(
            camera, gaussians, pipe, gaussian_normals_tensor
        )
        edge_features = pairwise_edge_features(
            image, inverse_depth, rendered_normals, observable
        )
        valid_owners = owner_ids[(owner_ids >= 0) & (owner_ids < num_gaussians)]
        visible_ids = torch.unique(valid_owners).cpu().numpy()
        view = {
            "view_index": local_view_index,
            "camera_index": camera_index,
            "name": record["img_name"],
            "image": image,
            "camera": camera,
            "masks": masks,
            "owner_ids": owner_ids.cpu(),
            "owner_weights": owner_weights.cpu(),
            "visible_ids": visible_ids,
        }
        views.append(view)
        for mask_index, mask_record in enumerate(masks):
            selected, membership, _, _ = lift_mask(
                mask_record["mask"],
                owner_ids,
                owner_weights,
                num_gaussians,
            )
            gaussian_ids = selected.nonzero(as_tuple=False).squeeze(1).cpu().numpy()
            if gaussian_ids.size == 0:
                continue
            fragments.append({
                "fragment_id": len(fragments),
                "view_index": local_view_index,
                "mask_index": mask_index,
                "sam_index": mask_record["sam_index"],
                "sam_score": mask_record["score"],
                "area_fraction": mask_record["area_fraction"],
                "gaussian_ids": gaussian_ids,
                "membership": membership[selected].half().cpu(),
                **mask_boundary_evidence(mask_record["mask"], edge_features),
            })
        print(
            f"{record['img_name']}: {len(masks)} masks, "
            f"{len(visible_ids)} visible Gaussians, "
            f"coverage={(coverage > 1e-3).float().mean().item():.3f}"
        )

    if observation_cache is None and args.observation_cache is not None:
        args.observation_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "source": str(args.source),
                "selected_camera_indices": selected_indices,
                "selected_camera_names": [view["name"] for view in views],
                "max_side": args.max_side,
                "resolution_divisor": args.resolution_divisor,
                "sam_model": args.sam_model,
                "points_per_batch": args.points_per_batch,
                "views": cached_views,
            },
            args.observation_cache,
        )

    fragments_by_view = {}
    for fragment in fragments:
        fragments_by_view.setdefault(fragment["view_index"], []).append(fragment)
    edges = []
    for first_view in range(len(views)):
        last_view = min(len(views), first_view + args.neighbor_window + 1)
        for second_view in range(first_view + 1, last_view):
            universe_size = np.union1d(
                views[first_view]["visible_ids"], views[second_view]["visible_ids"]
            ).size
            for first in fragments_by_view.get(first_view, []):
                for second in fragments_by_view.get(second_view, []):
                    edge = overlap_edge(first, second, universe_size)
                    if edge is not None:
                        edges.append(edge)

    significant_indices, p_cutoff = benjamini_hochberg(edges, args.fdr)
    accepted_indices = mutual_best_edges(edges, significant_indices)
    accepted_indices.sort(
        key=lambda index: (edges[index]["jaccard"], edges[index]["intersection"]),
        reverse=True,
    )
    union_find = DisjointViewUnionFind(fragments)
    merged_edges = []
    for edge_index in accepted_indices:
        edge = edges[edge_index]
        if union_find.merge(edge["first"], edge["second"]):
            merged_edges.append(edge)
    components = union_find.components()
    gaussian_xyz = gaussians.get_xyz.detach().cpu().numpy()
    gaussian_normals = gaussian_normals_tensor.cpu().numpy()
    gaussian_scaling = gaussian_scaling_tensor.cpu().numpy()
    gaussian_flatness = gaussian_scaling.min(axis=1) / np.maximum(
        gaussian_scaling.max(axis=1), 1e-12
    )
    component_records = evaluate_components(
        components,
        fragments,
        views,
        num_gaussians,
        device,
        gaussian_xyz,
        gaussian_normals,
        gaussian_flatness,
    )
    stable_components = [
        component for component in component_records if component["view_count"] >= 2
    ]
    hierarchy = hierarchy_candidates(stable_components, num_gaussians)

    write_component_visualization(
        args.output,
        stable_components,
        fragments,
        views,
        args.visualizations,
        num_gaussians,
        device,
        "support",
        "latent_objects.png",
    )
    write_component_visualization(
        args.output,
        stable_components,
        fragments,
        views,
        args.visualizations,
        num_gaussians,
        device,
        "volumetric",
        "latent_objects_volumetric.png",
    )
    write_component_visualization(
        args.output,
        stable_components,
        fragments,
        views,
        args.visualizations,
        num_gaussians,
        device,
        "normal_diversity",
        "latent_objects_normal_diversity.png",
    )
    write_component_visualization(
        args.output,
        stable_components,
        fragments,
        views,
        args.visualizations,
        num_gaussians,
        device,
        "inverse_depth_boundary",
        "latent_objects_depth_boundary.png",
    )
    write_component_visualization(
        args.output,
        stable_components,
        fragments,
        views,
        args.visualizations,
        num_gaussians,
        device,
        "appearance_boundary",
        "latent_objects_appearance_boundary.png",
    )
    removal_component = write_object_removal(
        args.output,
        stable_components,
        fragments,
        views,
        gaussians,
        pipe,
        background,
    )
    stable_loo = [
        component["mean_loo_iou"]
        for component in stable_components
        if component["mean_loo_iou"] is not None
    ]
    summary = {
        "source": str(args.source),
        "method": (
            "SAM2 fragments + FastGS top-2 support + hypergeometric FDR + "
            "mutual-best disjoint-view association"
        ),
        "selected_camera_indices": selected_indices,
        "selected_camera_names": [view["name"] for view in views],
        "num_gaussians": int(num_gaussians),
        "fragment_count": len(fragments),
        "tested_overlap_edge_count": len(edges),
        "significant_overlap_edge_count": len(significant_indices),
        "mutual_best_edge_count": len(accepted_indices),
        "merged_edge_count": len(merged_edges),
        "fdr": args.fdr,
        "fdr_p_value_cutoff": p_cutoff,
        "component_count": len(component_records),
        "stable_component_count": len(stable_components),
        "components_with_at_least_3_views": sum(
            component["view_count"] >= 3 for component in stable_components
        ),
        "mean_stable_component_loo_iou": (
            float(np.mean(stable_loo)) if stable_loo else None
        ),
        "median_stable_component_loo_iou": (
            float(np.median(stable_loo)) if stable_loo else None
        ),
        "removal_component": removal_component,
        "elapsed_seconds": time.perf_counter() - started,
        "components": [
            serializable_component(component) for component in stable_components
        ],
        "hierarchy_candidates": hierarchy,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    torch.save(
        {
            "source": str(args.source),
            "camera_indices": selected_indices,
            "resolution_divisor": args.resolution_divisor,
            "fragments": fragments,
            "components": [
                {
                    **serializable_component(component),
                    "gaussian_ids": torch.from_numpy(component["gaussian_ids"]),
                }
                for component in stable_components
            ],
            "merged_edges": merged_edges,
            "hierarchy_candidates": hierarchy,
        },
        args.output / "latent_object_map.pt",
    )
    print(json.dumps({
        key: value
        for key, value in summary.items()
        if key not in ("components", "hierarchy_candidates")
    }, indent=2))


if __name__ == "__main__":
    main()
