from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
from gsedit_scd.vendor.oscd.gaussian_renderer import render
from gsedit_scd.vendor.oscd.scene.gaussian_model import GaussianModel
from gsedit_scd.data.appearance import view_colors
from gsedit_scd.data.mask_geometry import find_image, labeled_panel, lift_mask, make_camera, owner_observation, resized_rgb
from gsedit_scd.data.membership import aggregate_membership, binary_iou, fit_threshold, render_membership
from gsedit_scd.data.edit_geometry import component_core_ids, evenly_spaced, multiview_observation_aware_translation, mask_image, mask_overlay, tensor_image, unpack_mask, write_rgb_batch

COLOR_PALETTE = (
    (0.90, 0.15, 0.18),
    (0.12, 0.72, 0.24),
    (0.12, 0.35, 0.92),
    (0.92, 0.72, 0.10),
    (0.82, 0.16, 0.82),
    (0.08, 0.76, 0.82),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate removal, colour, and full-clearance translation events "
            "for every reviewed FastGS semantic instance."
        )
    )
    parser.add_argument("--selected-objects", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--views-per-object",
        type=int,
        default=3,
        help="Use zero for every view in the stable anchor track.",
    )
    parser.add_argument("--minimum-valid-translation-views", type=int, default=2)
    parser.add_argument(
        "--minimum-translation-visible-retention", type=float, default=0.5
    )
    parser.add_argument(
        "--minimum-background-alpha",
        type=float,
        help="Mark low-coverage disoccluded object pixels as ignore.",
    )
    parser.add_argument(
        "--minimum-background-provenance",
        type=float,
        help=(
            "Require remaining responsibility to come from Gaussians observed "
            "outside this instance in the reference views."
        ),
    )
    parser.add_argument("--translation-gap-extents", type=float, default=0.5)
    parser.add_argument("--translation-direction-attempts", type=int, default=24)
    parser.add_argument(
        "--maximum-translation-seed-views",
        type=int,
        default=0,
        help=(
            "Bound observation-aware translation search to evenly spaced "
            "stable views; zero keeps the exhaustive search."
        ),
    )
    parser.add_argument("--colour-mix", type=float, default=0.45)
    parser.add_argument("--minimum-semantic-loo-iou", type=float, default=0.65)
    parser.add_argument("--minimum-fold-iou", type=float, default=0.5)
    parser.add_argument("--minimum-moved-area-ratio", type=float, default=0.5)
    parser.add_argument("--maximum-old-new-overlap", type=float, default=0.01)
    parser.add_argument("--causal-effect-mass", type=float, default=0.99)
    parser.add_argument("--max-side", type=int, default=1008)
    parser.add_argument(
        "--resolution-divisor", type=int, default=0,
        help="Match O-SCD by independently rounding width and height",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--export-diagnostics", action="store_true",
                        help="Also save background diagnostic maps and workflow montage images.")
    return parser.parse_args()


def resolve_review(review_path):
    review = json.loads(review_path.read_text())
    candidate_path = Path(review["candidate_manifest"])
    candidate_manifest = json.loads(candidate_path.read_text())
    object_prefix = candidate_manifest.get("candidate_prefix", "cantina")
    candidates = {
        candidate["candidate_id"]: candidate
        for candidate in candidate_manifest["candidates"]
    }
    resolved = []
    seen = set()
    for index, selection in enumerate(review["objects"], start=1):
        candidate_id = selection["candidate_id"]
        if candidate_id in seen:
            raise ValueError(f"Duplicate reviewed candidate: {candidate_id}")
        if candidate_id not in candidates:
            raise ValueError(f"Unknown reviewed candidate: {candidate_id}")
        seen.add(candidate_id)
        candidate = dict(candidates[candidate_id])
        anchor = int(
            selection.get(
                "anchor_component_id_override",
                candidate["anchor_component_id"],
            )
        )
        if anchor not in candidate["association_component_ids"]:
            raise ValueError(f"{candidate_id}: anchor override is outside its group")
        candidate.update({
            "object_id": f"{object_prefix}_object_{index:03d}",
            "name": selection["name"],
            "anchor_component_id": anchor,
        })
        resolved.append(candidate)
    return review, candidate_manifest, resolved


def accepted_association_edge(edge, criteria):
    return (
        float(edge["child_containment"])
        >= float(criteria["minimum_child_containment"])
        and float(edge["jaccard"])
        >= float(criteria["minimum_edge_jaccard"])
        and float(edge["p_value"])
        <= float(criteria["maximum_edge_p_value"])
    )


def select_object_fragments(
    candidate,
    components,
    fragments,
    hierarchy_edges,
    criteria,
    count,
):
    anchor = components[candidate["anchor_component_id"]]
    by_view = {}
    for fragment_id in anchor["fragment_ids"]:
        fragment = fragments[fragment_id]
        score = (
            float(fragment["sam_score"]),
            float(fragment["area_fraction"]),
        )
        view_index = int(fragment["view_index"])
        if view_index not in by_view or score > by_view[view_index][0]:
            by_view[view_index] = (
                score,
                fragment_id,
                candidate["anchor_component_id"],
            )

    required_count = 2 if count == 0 else count
    anchor_views = sorted(by_view)
    selected_components = {candidate["anchor_component_id"]}
    if len(by_view) < required_count:
        group = set(int(value) for value in candidate["association_component_ids"])
        graph_edges = []
        for edge in hierarchy_edges:
            child = int(edge["child_component"])
            parent = int(edge["parent_component"])
            if child not in group or parent not in group:
                continue
            if not accepted_association_edge(edge, criteria):
                continue
            graph_edges.append((
                float(edge["jaccard"]),
                float(edge["child_containment"]),
                child,
                parent,
            ))
        while len(by_view) < required_count:
            frontier = [
                edge for edge in graph_edges
                if (edge[2] in selected_components)
                != (edge[3] in selected_components)
            ]
            if not frontier:
                break
            _, _, child, parent = max(frontier)
            component_id = (
                parent if child in selected_components else child
            )
            selected_components.add(component_id)
            for fragment_id in components[component_id]["fragment_ids"]:
                fragment = fragments[fragment_id]
                view_index = int(fragment["view_index"])
                if view_index in by_view:
                    continue
                score = (
                    float(fragment["sam_score"]),
                    float(fragment["area_fraction"]),
                )
                by_view[view_index] = (score, fragment_id, component_id)
    if len(by_view) < required_count:
        raise ValueError(
            f"{candidate['object_id']}: strong anchor-connected track has only "
            f"{len(by_view)} independent views"
        )
    if count == 0:
        selected_views = sorted(by_view)
    elif len(anchor_views) >= count:
        selected_views = evenly_spaced(anchor_views, count)
    else:
        novel_views = sorted(
            (view_index for view_index in by_view if view_index not in anchor_views),
            key=lambda view_index: by_view[view_index][0],
            reverse=True,
        )
        selected_views = anchor_views + novel_views[: count - len(anchor_views)]
    selected_fragments = [
        fragments[by_view[view_index][1]] for view_index in selected_views
    ]
    supporting_components = sorted({
        int(by_view[view_index][2]) for view_index in selected_views
    })
    return selected_fragments, supporting_components


def aggregate_background_reliability(views, gaussian_count):
    background_mass = torch.zeros(gaussian_count, device="cuda")
    total_mass = torch.zeros_like(background_mass)
    for view in views:
        _, _, positive, total = lift_mask(
            torch.from_numpy(view["sam_mask"]),
            view["owner_ids"],
            view["owner_weights"],
            gaussian_count,
        )
        background_mass += (total - positive).clamp_min(0.0)
        total_mass += total
    reliability = background_mass / total_mass.clamp_min(1e-8)
    reliability[total_mass <= 0.0] = 0.0
    return reliability.clamp(0.0, 1.0)


def choose_colour_target(colors, object_ids):
    source = colors[object_ids].median(dim=0).values
    palette = torch.tensor(COLOR_PALETTE, device=colors.device, dtype=colors.dtype)
    index = torch.linalg.vector_norm(palette - source[None], dim=1).argmax()
    return palette[index], source






def hard_mask(score, threshold):
    return (score >= threshold).cpu().numpy()


def overlap_fraction(first, second):
    intersection = int(np.logical_and(first, second).sum())
    denominator = min(int(first.sum()), int(second.sum()))
    return intersection / denominator if denominator else 0.0


def effect_localization(effect, mask):
    positive = torch.from_numpy(mask).to(effect.device).bool()
    total = effect.sum().clamp_min(1e-12)
    return float((effect[positive].sum() / total).item())


def effect_mass_mask(effect, retained_mass):
    flat = effect.reshape(-1)
    total = flat.sum()
    if float(total.item()) <= 0.0:
        return np.zeros(tuple(effect.shape), dtype=bool)
    sorted_values = torch.sort(flat, descending=True).values
    cumulative = torch.cumsum(sorted_values, dim=0)
    index = int(torch.searchsorted(cumulative, retained_mass * total).item())
    threshold = sorted_values[min(index, sorted_values.numel() - 1)]
    return (effect >= threshold).cpu().numpy()


def partition_event_masks(candidate_positive, effect_support, invalid_interior=None):
    candidate_positive = np.asarray(candidate_positive, dtype=bool)
    effect_support = np.asarray(effect_support, dtype=bool)
    if candidate_positive.shape != effect_support.shape:
        raise ValueError("Candidate and effect masks have different shapes")
    if invalid_interior is None:
        invalid_interior = np.zeros_like(candidate_positive)
    else:
        invalid_interior = np.asarray(invalid_interior, dtype=bool)
        if invalid_interior.shape != candidate_positive.shape:
            raise ValueError("Interior-validity mask has a different shape")
    invalid_interior = np.logical_and(invalid_interior, candidate_positive)
    positive = np.logical_and(candidate_positive, ~invalid_interior)
    outer_ignore = np.logical_and(effect_support, ~candidate_positive)
    ignore = np.logical_or(invalid_interior, outer_ignore)
    return positive, np.logical_and(ignore, ~positive)


def write_soft_map(path, score):
    if torch.is_tensor(score):
        score = score.detach().float().cpu().numpy()
    array = np.clip(np.asarray(score, dtype=np.float32), 0.0, 1.0)
    encoded = np.rint(array * 65535.0).astype(np.uint16)
    Image.fromarray(encoded, mode="I;16").save(path, compress_level=1)


def positive_ignore_image(positive, ignore):
    image = np.zeros((*positive.shape, 3), dtype=np.uint8)
    image[ignore] = np.asarray((255, 120, 40), dtype=np.uint8)
    image[positive] = 255
    return image


@torch.no_grad()
def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.views_per_object < 0:
        raise ValueError("views-per-object must be non-negative")
    if args.minimum_valid_translation_views < 1:
        raise ValueError("minimum-valid-translation-views must be positive")
    if not 0.0 <= args.minimum_translation_visible_retention <= 1.0:
        raise ValueError(
            "minimum-translation-visible-retention must lie in [0, 1]"
        )
    if (
        args.minimum_background_alpha is not None
        and not 0.0 <= args.minimum_background_alpha <= 1.0
    ):
        raise ValueError("minimum-background-alpha must lie in [0, 1]")
    if (
        args.minimum_background_provenance is not None
        and not 0.0 <= args.minimum_background_provenance <= 1.0
    ):
        raise ValueError("minimum-background-provenance must lie in [0, 1]")
    if args.translation_gap_extents < 0.0:
        raise ValueError("translation-gap-extents must be non-negative")
    if not 0.0 < args.colour_mix <= 1.0:
        raise ValueError("colour-mix must lie in (0, 1]")
    if not 0.0 < args.causal_effect_mass <= 1.0:
        raise ValueError("causal-effect-mass must lie in (0, 1]")
    if args.resolution_divisor < 0:
        raise ValueError("--resolution-divisor must be zero or positive")
    review, candidate_manifest, candidates = resolve_review(args.selected_objects)
    source = Path(candidate_manifest["source"])
    object_map_path = Path(candidate_manifest["object_map"])
    cache_path = Path(candidate_manifest["observation_cache"])
    latent_map = torch.load(object_map_path, map_location="cpu", weights_only=False)
    observation_cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    components = {
        int(component["component_id"]): component
        for component in latent_map["components"]
    }
    fragments = latent_map["fragments"]
    camera_records = json.loads(
        (source / "reference_reconstruction" / "cameras.json").read_text()
    )
    expected_names = [
        camera_records[index]["img_name"] for index in latent_map["camera_indices"]
    ]
    if expected_names != [view["name"] for view in observation_cache["views"]]:
        raise ValueError("Object map and observation cache camera names differ")
    if int(observation_cache["max_side"]) != args.max_side:
        raise ValueError("--max-side does not match observation cache")
    if int(observation_cache.get("resolution_divisor", 0)) != args.resolution_divisor:
        raise ValueError(
            "--resolution-divisor does not match observation cache"
        )

    model_path = (
        source
        / "reference_reconstruction"
        / "point_cloud"
        / "iteration_30000"
        / "point_cloud.ply"
    )
    gaussians = GaussianModel(3, 3)
    gaussians.load_ply(str(model_path))
    gaussian_count = gaussians.get_xyz.shape[0]
    points = gaussians.get_xyz.detach()
    opacity = gaussians.get_opacity.detach()
    pipe = SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
    )
    background = torch.zeros(3, device="cuda")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    view_cache = {}

    def load_view(fragment):
        view_index = int(fragment["view_index"])
        if view_index not in view_cache:
            camera_index = latent_map["camera_indices"][view_index]
            record = camera_records[camera_index]
            image = resized_rgb(
                find_image(source, record["img_name"]),
                args.max_side,
                args.resolution_divisor,
            )
            camera = make_camera(record, image)
            owner_ids, owner_weights, coverage, baseline = owner_observation(
                camera, gaussians, pipe, background
            )
            view_cache[view_index] = {
                "view_index": view_index,
                "camera_index": camera_index,
                "record": record,
                "image": np.asarray(image),
                "camera": camera,
                "owner_ids": owner_ids,
                "owner_weights": owner_weights,
                "coverage": coverage,
                "render": baseline,
            }
        view = dict(view_cache[view_index])
        view["fragment"] = fragment
        view["sam_mask"] = unpack_mask(
            observation_cache["views"][view_index]["masks"][
                int(fragment["mask_index"])
            ]
        )
        return view

    args.output.mkdir(parents=True)
    object_root = args.output / "objects"
    object_root.mkdir()
    accepted_objects = []
    rejected_objects = []
    sample_records = []
    for candidate_index, candidate in enumerate(candidates):
        try:
            selected_fragments, supporting_components = select_object_fragments(
                candidate,
                components,
                fragments,
                latent_map.get("hierarchy_candidates", ()),
                candidate_manifest["criteria"],
                args.views_per_object,
            )
            views = [load_view(fragment) for fragment in selected_fragments]
            anchor = components[candidate["anchor_component_id"]]
            object_ids = anchor["gaussian_ids"].long().cuda()
            sam_targets = [
                torch.from_numpy(view["sam_mask"]).cuda() for view in views
            ]
            fold_records = []
            semantic_heldout_ious = []
            for heldout in range(len(views)):
                training = [
                    index for index in range(len(views)) if index != heldout
                ]
                membership, _ = aggregate_membership(
                    views, training, object_ids, gaussian_count
                )
                semantic_maps = [
                    render_membership(
                        view["camera"], gaussians, pipe, background, membership
                    )
                    for view in views
                ]
                semantic_threshold, training_iou = fit_threshold(
                    semantic_maps, sam_targets, training
                )
                semantic_iou = binary_iou(
                    semantic_maps[heldout] >= semantic_threshold,
                    sam_targets[heldout],
                )
                semantic_heldout_ious.append(semantic_iou)
                fold_records.append({
                    "heldout_camera": views[heldout]["record"]["img_name"],
                    "semantic_threshold": semantic_threshold,
                    "training_mean_iou": training_iou,
                    "semantic_heldout_iou": semantic_iou,
                })
            mean_semantic_iou = float(np.mean(semantic_heldout_ious))
            minimum_semantic_iou = float(np.min(semantic_heldout_ious))
            if (
                mean_semantic_iou < args.minimum_semantic_loo_iou
                or minimum_semantic_iou < args.minimum_fold_iou
            ):
                raise ValueError(
                    f"semantic LOO {mean_semantic_iou:.3f}/{minimum_semantic_iou:.3f}"
                )

            all_indices = list(range(len(views)))
            membership, observed = aggregate_membership(
                views, all_indices, object_ids, gaussian_count
            )
            background_reliability = aggregate_background_reliability(
                views, gaussian_count
            )
            semantic_maps = [
                render_membership(
                    view["camera"], gaussians, pipe, background, membership
                )
                for view in views
            ]
            semantic_threshold, semantic_training_iou = fit_threshold(
                semantic_maps, sam_targets, all_indices
            )
            ownership = membership[object_ids].clamp(0.0, 1.0)
            motion_ids = object_ids[ownership >= 0.5]
            core_ids = torch.from_numpy(
                component_core_ids(anchor, fragments)
            ).long().cuda()
            selected_translation = None
            translation_report = None
            valid_translation_views = set()
            moved_points = None
            moved_maps = None
            moved_masks = None
            motion_source_maps = None
            motion_threshold = None
            motion_source_iou = None
            if motion_ids.numel() > 0:
                motion_membership = torch.zeros_like(membership)
                motion_membership[motion_ids] = membership[motion_ids]
                motion_source_maps = [
                    render_membership(
                        view["camera"],
                        gaussians,
                        pipe,
                        background,
                        motion_membership,
                    )
                    for view in views
                ]
                motion_threshold, motion_source_iou = fit_threshold(
                    motion_source_maps, sam_targets, all_indices
                )
                motion_core_ids = core_ids[torch.isin(core_ids, motion_ids)]
                if motion_core_ids.numel() == 0:
                    motion_core_ids = motion_ids
                if len(views) >= args.minimum_valid_translation_views:
                    translation_search_started = time.perf_counter()
                    translation, translation_report = (
                        multiview_observation_aware_translation(
                            points,
                            motion_core_ids,
                            [view["camera"] for view in views],
                            gap_extents=args.translation_gap_extents,
                            direction_attempts=(
                                args.translation_direction_attempts
                            ),
                            minimum_valid_views=(
                                args.minimum_valid_translation_views
                            ),
                            minimum_visible_retention=(
                                args.minimum_translation_visible_retention
                            ),
                            maximum_old_new_overlap=(
                                args.maximum_old_new_overlap
                            ),
                            maximum_seed_views=(
                                args.maximum_translation_seed_views or None
                            ),
                            generator=generator,
                        )
                    )
                    translation_report["search_seconds"] = (
                        time.perf_counter() - translation_search_started
                    )
                else:
                    translation = None
                    translation_report = {
                        "translation_available": False,
                        "reason": "object_has_too_few_reference_views",
                        "minimum_valid_views": (
                            args.minimum_valid_translation_views
                        ),
                    }
                if translation is not None:
                    candidate_points = points.clone()
                    candidate_points[motion_ids] += translation
                    candidate_maps = [
                        render_membership(
                            view["camera"],
                            gaussians,
                            pipe,
                            background,
                            motion_membership,
                            override_xyz=candidate_points,
                        )
                        for view in views
                    ]
                    candidate_masks = [
                        hard_mask(score, motion_threshold)
                        for score in candidate_maps
                    ]
                    area_ratios = [
                        int(moved.sum())
                        / max(int(view["sam_mask"].sum()), 1)
                        for moved, view in zip(candidate_masks, views)
                    ]
                    overlaps = [
                        overlap_fraction(view["sam_mask"], moved)
                        for moved, view in zip(candidate_masks, views)
                    ]
                    geometry_valid = set(
                        translation_report["selected"][
                            "valid_view_indices"
                        ]
                    )
                    renderer_valid = [
                        index
                        for index, (moved, overlap) in enumerate(
                            zip(candidate_masks, overlaps)
                        )
                        if (
                            index in geometry_valid
                            and bool(moved.any())
                            and overlap < args.maximum_old_new_overlap
                        )
                    ]
                    translation_report["renderer_valid_view_indices"] = (
                        renderer_valid
                    )
                    if (
                        len(renderer_valid)
                        >= args.minimum_valid_translation_views
                    ):
                        selected_translation = {
                            "translation": translation,
                            "moved_area_ratios": area_ratios,
                            "old_new_overlap_fractions": overlaps,
                            "valid_view_indices": renderer_valid,
                            "observation_report": translation_report,
                        }
                        moved_points = candidate_points
                        moved_maps = candidate_maps
                        moved_masks = candidate_masks
                        valid_translation_views = set(renderer_valid)

            removed_opacity = opacity.clone()
            removed_opacity[object_ids] *= 1.0 - ownership[:, None]
            anchor_colors = view_colors(gaussians, views[0]["camera"])
            colour_target, source_colour = choose_colour_target(
                anchor_colors,
                motion_ids if motion_ids.numel() else object_ids,
            )
            object_directory = object_root / (
                f"{candidate['object_id']}_{candidate['name']}"
            )
            object_directory.mkdir()
            object_samples = []
            translation_available = selected_translation is not None
            if translation_available and not valid_translation_views:
                valid_translation_views = set(range(len(views)))
            for view_index, view in enumerate(views):
                translation_view_available = (
                    translation_available
                    and view_index in valid_translation_views
                )
                standard_render = view["render"]
                reference = view_cache[view["view_index"]]
                if "edit_colors" not in reference:
                    reference["edit_colors"] = view_colors(gaussians, view["camera"])
                    reference["edit_baseline"] = render(
                        view["camera"], gaussians, pipe, background,
                        override_color=reference["edit_colors"].contiguous(),
                    )["render"]
                colors = reference["edit_colors"]
                baseline = reference["edit_baseline"]
                removed_result = render(
                    view["camera"],
                    gaussians,
                    pipe,
                    background,
                    override_color=colors.contiguous(),
                    override_opacity=removed_opacity,
                    return_ray_stats=True,
                )
                removed = removed_result["render"]
                removed_ray_stats = removed_result["ray_stats"]
                if removed_ray_stats.ndim != 3 or removed_ray_stats.shape[0] < 1:
                    raise ValueError("FastGS did not return alpha coverage")
                removed_alpha = removed_ray_stats[0].clamp(0.0, 1.0)
                background_colors = background_reliability[:, None].expand(
                    -1, 3
                ).contiguous()
                background_weighted = render(
                    view["camera"],
                    gaussians,
                    pipe,
                    background,
                    override_color=background_colors,
                    override_opacity=removed_opacity,
                )["render"].mean(dim=0)
                background_provenance = (
                    background_weighted
                    / removed_alpha.clamp_min(1e-6)
                ).clamp(0.0, 1.0)
                background_provenance[removed_alpha <= 1e-6] = 0.0
                translated = None
                if translation_view_available:
                    translated = render(
                        view["camera"],
                        gaussians,
                        pipe,
                        background,
                        override_color=colors.contiguous(),
                        override_xyz=moved_points,
                    )["render"]
                altered_colors = colors.clone()
                altered_colors[object_ids] = (
                    colors[object_ids]
                    + args.colour_mix
                    * ownership[:, None]
                    * (colour_target - colors[object_ids])
                )
                recoloured = render(
                    view["camera"],
                    gaussians,
                    pipe,
                    background,
                    override_color=altered_colors.contiguous(),
                )["render"]
                sam = view["sam_mask"]
                object_causal_score = semantic_maps[view_index].clamp(0.0, 1.0)
                translation_causal_score = None
                if translation_view_available:
                    old_motion_score = motion_source_maps[view_index].clamp(0.0, 1.0)
                    new_motion_score = moved_maps[view_index].clamp(0.0, 1.0)
                    translation_causal_score = 1.0 - (
                        (1.0 - old_motion_score) * (1.0 - new_motion_score)
                    )
                moved_mask = (
                    moved_masks[view_index]
                    if translation_view_available
                    else np.zeros_like(sam)
                )
                removal_effect = (removed - baseline).abs().mean(dim=0)
                colour_effect = (recoloured - baseline).abs().mean(dim=0)
                translation_effect = (
                    (translated - baseline).abs().mean(dim=0)
                    if translation_view_available
                    else None
                )
                removal_support = effect_mass_mask(
                    removal_effect, args.causal_effect_mass
                )
                colour_support = effect_mass_mask(
                    colour_effect, args.causal_effect_mass
                )
                background_invalid = np.zeros_like(sam)
                if args.minimum_background_alpha is not None:
                    background_invalid = (
                        removed_alpha.cpu().numpy()
                        < args.minimum_background_alpha
                    )
                if args.minimum_background_provenance is not None:
                    background_invalid = np.logical_or(
                        background_invalid,
                        background_provenance.cpu().numpy()
                        < args.minimum_background_provenance,
                    )
                removal_positive, removal_ignore = partition_event_masks(
                    sam,
                    removal_support,
                    background_invalid,
                )
                colour_positive, colour_ignore = partition_event_masks(
                    sam,
                    colour_support,
                )
                removal_envelope = np.logical_or(
                    removal_positive, removal_ignore
                )
                colour_envelope = np.logical_or(
                    colour_positive, colour_ignore
                )
                geometric_translation_union = (
                    np.logical_or(sam, moved_mask)
                    if translation_view_available
                    else np.zeros_like(sam)
                )
                translation_support = (
                    effect_mass_mask(
                        translation_effect,
                        args.causal_effect_mass,
                    )
                    if translation_view_available
                    else np.zeros_like(sam)
                )
                translation_invalid = np.logical_and(
                    sam, background_invalid
                )
                (
                    translation_positive,
                    translation_ignore,
                ) = partition_event_masks(
                    geometric_translation_union,
                    translation_support,
                    translation_invalid,
                )
                translation_envelope = np.logical_or(
                    translation_positive,
                    translation_ignore,
                )
                identity_error = float(
                    (baseline - standard_render).abs().mean().item()
                )
                stem = view["record"]["img_name"]
                files = {
                    "reference": f"{stem}_reference.png",
                    "removed": f"{stem}_removed.png",
                    "recoloured": f"{stem}_recoloured.png",
                    "translated": (
                        f"{stem}_translated.png" if translation_view_available else None
                    ),
                    "object_mask": f"{stem}_object_mask.png",
                    "background_coverage_map": (
                        f"{stem}_background_coverage_map.png"
                        if args.export_diagnostics and args.minimum_background_alpha is not None else None
                    ),
                    "background_provenance_map": (
                        f"{stem}_background_provenance_map.png"
                        if args.export_diagnostics and args.minimum_background_provenance is not None else None
                    ),
                    "removal_target_map": (
                        f"{stem}_removal_target_map.png"
                    ),
                    "removal_positive_mask": f"{stem}_removal_positive_mask.png",
                    "removal_ignore_mask": f"{stem}_removal_ignore_mask.png",
                    "colour_target_map": (
                        f"{stem}_colour_target_map.png"
                    ),
                    "colour_positive_mask": f"{stem}_colour_positive_mask.png",
                    "colour_ignore_mask": f"{stem}_colour_ignore_mask.png",
                    "translation_target_map": (
                        f"{stem}_translation_target_map.png"
                        if translation_view_available
                        else None
                    ),
                    "translation_positive_mask": (
                        f"{stem}_translation_positive_mask.png"
                        if translation_view_available else None
                    ),
                    "translation_ignore_mask": (
                        f"{stem}_translation_ignore_mask.png"
                        if translation_view_available else None
                    ),
                    "workflow": f"{stem}_workflow.png" if args.export_diagnostics else None,
                }
                rgb_writes = []
                reference_path = object_directory / files["reference"]
                if "edit_reference_path" not in reference:
                    rgb_writes.append((reference_path, tensor_image(baseline)))
                    reference["edit_reference_path"] = reference_path
                else:
                    shutil.copyfile(reference["edit_reference_path"], reference_path)
                rgb_writes.append((object_directory / files["removed"], tensor_image(removed)))
                rgb_writes.append((
                    object_directory / files["recoloured"], tensor_image(recoloured)
                ))
                rgb_writes.append((object_directory / files["object_mask"], mask_image(sam)))
                if files["background_coverage_map"] is not None:
                    write_soft_map(
                        object_directory / files["background_coverage_map"],
                        removed_alpha,
                    )
                if files["background_provenance_map"] is not None:
                    write_soft_map(
                        object_directory / files["background_provenance_map"],
                        background_provenance,
                    )
                write_soft_map(
                    object_directory / files["removal_target_map"],
                    object_causal_score,
                )
                shutil.copyfile(object_directory / files["removal_target_map"],
                                object_directory / files["colour_target_map"])
                rgb_writes.append((
                    object_directory / files["removal_positive_mask"],
                    mask_image(removal_positive),
                ))
                rgb_writes.append((
                    object_directory / files["removal_ignore_mask"],
                    mask_image(removal_ignore),
                ))
                rgb_writes.append((
                    object_directory / files["colour_positive_mask"],
                    mask_image(colour_positive),
                ))
                rgb_writes.append((
                    object_directory / files["colour_ignore_mask"],
                    mask_image(colour_ignore),
                ))
                if translation_view_available:
                    write_soft_map(
                        object_directory / files["translation_target_map"],
                        translation_causal_score,
                    )
                    rgb_writes.append((
                        object_directory / files["translated"],
                        tensor_image(translated),
                    ))
                    rgb_writes.append((
                        object_directory / files["translation_positive_mask"],
                        mask_image(translation_positive),
                    ))
                    rgb_writes.append((
                        object_directory / files["translation_ignore_mask"],
                        mask_image(translation_ignore),
                    ))
                if args.export_diagnostics:
                    translated_panel = (
                        tensor_image(translated)
                        if translation_view_available
                        else np.zeros_like(tensor_image(baseline))
                    )
                    first_row = np.concatenate((
                        labeled_panel(mask_overlay(view["image"], sam), "reference SAM"),
                        labeled_panel(tensor_image(baseline), "matched FastGS H0"),
                        labeled_panel(tensor_image(removed), "removed"),
                        labeled_panel(tensor_image(recoloured), "recoloured"),
                        labeled_panel(
                            translated_panel,
                            "translated" if translation_view_available
                            else "translation unavailable",
                        ),
                    ), axis=1)
                    second_row = np.concatenate((
                        labeled_panel(mask_image(sam), "object SAM"),
                        labeled_panel(
                            positive_ignore_image(removal_positive, removal_ignore),
                            "removal: white + ignore",
                        ),
                        labeled_panel(
                            positive_ignore_image(colour_positive, colour_ignore),
                            "colour: white + ignore",
                        ),
                        labeled_panel(mask_image(moved_mask), "moved mask"),
                        labeled_panel(
                            positive_ignore_image(
                                translation_positive,
                                translation_ignore,
                            ),
                            "translation: white + ignore",
                        ),
                    ), axis=1)
                    rgb_writes.append((
                        object_directory / files["workflow"],
                        np.concatenate((first_row, second_row), axis=0),
                    ))
                write_rgb_batch(rgb_writes)
                sample = {
                    "object_id": candidate["object_id"],
                    "name": candidate["name"],
                    "camera_name": stem,
                    "view_index": int(view["view_index"]),
                    "files": files,
                    "available_events": (
                        ["removed", "recoloured", "translated"]
                        if translation_view_available
                        else ["removed", "recoloured"]
                    ),
                    "identity_render_mean_absolute_error": identity_error,
                    "background_coverage_mean_in_object": float(
                        removed_alpha[
                            torch.from_numpy(sam).to(removed_alpha.device)
                        ].mean().item()
                    ),
                    "background_provenance_mean_in_object": float(
                        background_provenance[
                            torch.from_numpy(sam).to(
                                background_provenance.device
                            )
                        ].mean().item()
                    ),
                    "background_invalid_fraction_of_object": (
                        int(np.logical_and(background_invalid, sam).sum())
                        / max(int(sam.sum()), 1)
                    ),
                    "removal_effect_localization": effect_localization(
                        removal_effect, sam
                    ),
                    "removal_effect_localization_in_training_envelope": (
                        effect_localization(removal_effect, removal_envelope)
                    ),
                    "removal_ignore_area_ratio_to_object": (
                        int(removal_ignore.sum()) / max(int(sam.sum()), 1)
                    ),
                    "colour_effect_localization": effect_localization(
                        colour_effect, sam
                    ),
                    "colour_effect_localization_in_training_envelope": (
                        effect_localization(colour_effect, colour_envelope)
                    ),
                    "colour_ignore_area_ratio_to_object": (
                        int(colour_ignore.sum()) / max(int(sam.sum()), 1)
                    ),
                    "translation_effect_localization_in_object_union": (
                        effect_localization(
                            translation_effect,
                            geometric_translation_union,
                        )
                        if translation_view_available else None
                    ),
                    "translation_effect_localization_in_training_envelope": (
                        effect_localization(
                            translation_effect,
                            translation_envelope,
                        )
                        if translation_view_available else None
                    ),
                    "translation_ignore_area_ratio_to_object": (
                        int(translation_ignore.sum()) / max(int(sam.sum()), 1)
                        if translation_view_available else None
                    ),
                    "old_new_overlap_fraction": (
                        overlap_fraction(sam, moved_mask)
                        if translation_view_available else None
                    ),
                    "moved_area_ratio": (
                        int(moved_mask.sum()) / max(int(sam.sum()), 1)
                        if translation_view_available else None
                    ),
                }
                object_samples.append(sample)
                sample_records.append(sample)

            translation_record = None
            if translation_available and True:
                valid_indices = selected_translation["valid_view_indices"]
                translation_record = {
                    "protocol": (
                        "binary_remove_add_bounded_seed_per_view_v2"
                        if args.maximum_translation_seed_views
                        else "binary_remove_add_per_view_v1"
                    ),
                    "final_translation": selected_translation[
                        "translation"
                    ].cpu().tolist(),
                    "motion_membership_threshold": 0.5,
                    "motion_source_threshold": motion_threshold,
                    "motion_source_mean_iou": motion_source_iou,
                    "moved_area_ratios": selected_translation[
                        "moved_area_ratios"
                    ],
                    "old_new_overlap_fractions": selected_translation[
                        "old_new_overlap_fractions"
                    ],
                    "valid_view_indices": valid_indices,
                    "valid_cameras": [
                        views[index]["record"]["img_name"]
                        for index in valid_indices
                    ],
                    "rejected_cameras": [
                        view["record"]["img_name"]
                        for index, view in enumerate(views)
                        if index not in valid_translation_views
                    ],
                    "observation_report": selected_translation[
                        "observation_report"
                    ],
                }
            elif translation_available:
                translation_record = {
                    "protocol": "legacy_all_selected_views_v5",
                    "initial_translation": selected_translation[
                        "initial_translation"
                    ].cpu().tolist(),
                    "final_translation": selected_translation[
                        "translation"
                    ].cpu().tolist(),
                    "full_object_clearance_scale": selected_translation["scale"],
                    "border_margin": selected_translation["border_margin"],
                    "per_view_clearance": selected_translation["clearance"],
                    "motion_membership_threshold": 0.5,
                    "motion_source_threshold": motion_threshold,
                    "motion_source_mean_iou": motion_source_iou,
                    "moved_area_ratios": selected_translation[
                        "moved_area_ratios"
                    ],
                    "fully_visible_view_count": sum(
                        ratio >= args.minimum_moved_area_ratio
                        for ratio in selected_translation["moved_area_ratios"]
                    ),
                    "old_new_overlap_fractions": selected_translation[
                        "old_new_overlap_fractions"
                    ],
                }
            accepted_objects.append({
                "object_id": candidate["object_id"],
                "name": candidate["name"],
                "candidate_id": candidate["candidate_id"],
                "anchor_component_id": candidate["anchor_component_id"],
                "selected_supporting_component_ids": supporting_components,
                "association_component_ids": candidate[
                    "association_component_ids"
                ],
                "gaussian_count": int(object_ids.numel()),
                "majority_owned_gaussian_count": int(motion_ids.numel()),
                "mean_semantic_ownership": float(ownership.mean().item()),
                "observed_gaussian_count": int(observed.sum().item()),
                "selected_cameras": [
                    view["record"]["img_name"] for view in views
                ],
                "semantic_threshold": semantic_threshold,
                "semantic_training_mean_iou": semantic_training_iou,
                "semantic_leave_one_out": fold_records,
                "mean_semantic_heldout_iou": mean_semantic_iou,
                "minimum_semantic_heldout_iou": minimum_semantic_iou,
                "translation": translation_record,
                "colour": {
                    "source_median_rgb_in_anchor_view": source_colour.cpu().tolist(),
                    "target_rgb": colour_target.cpu().tolist(),
                    "mix": args.colour_mix,
                },
                "samples": object_samples,
            })
            print(
                f"[{candidate_index + 1}/{len(candidates)}] "
                f"accepted {candidate['object_id']} {candidate['name']} "
                f"({'3 events' if translation_available else '2 events'})"
            )
        except (RuntimeError, ValueError) as error:
            rejected_objects.append({
                "object_id": candidate["object_id"],
                "name": candidate["name"],
                "candidate_id": candidate["candidate_id"],
                "reason": str(error),
            })
            print(
                f"[{candidate_index + 1}/{len(candidates)}] "
                f"rejected {candidate['object_id']} {candidate['name']}: {error}"
            )

    translation_object_count = sum(
        record["translation"] is not None for record in accepted_objects
    )
    event_sample_count = sum(
        len(sample["available_events"]) for sample in sample_records
    )
    manifest = {
        "semantic_review": str(args.selected_objects),
        "review_type": review["review_type"],
        "source": str(source),
        "object_map": str(object_map_path),
        "observation_cache": str(cache_path),
        "seed": args.seed,
        "settings": {
            "views_per_object": args.views_per_object,
            "view_selection": (
                "all_stable_anchor_track_views"
                if args.views_per_object == 0 else "fixed_evenly_spaced_views"
            ),
            "translation_protocol": (
                "observation_aware_bounded_seed_per_view_v2"
                if (
                    True
                    and args.maximum_translation_seed_views
                )
                else "observation_aware_per_view_v1"
            ),
            "minimum_valid_translation_views": (
                args.minimum_valid_translation_views
            ),
            "minimum_translation_visible_retention": (
                args.minimum_translation_visible_retention
            ),
            "minimum_background_alpha": args.minimum_background_alpha,
            "minimum_background_provenance": (
                args.minimum_background_provenance
            ),
            "background_validity": (
                "remaining_coverage_and_cross_view_non_instance_provenance"
            ),
            "translation_gap_extents": args.translation_gap_extents,
            "translation_direction_attempts": args.translation_direction_attempts,
            "maximum_translation_seed_views": (
                args.maximum_translation_seed_views or None
            ),
            "colour_mix": args.colour_mix,
            "minimum_semantic_loo_iou": args.minimum_semantic_loo_iou,
            "minimum_fold_iou": args.minimum_fold_iou,
            "full_visibility_moved_area_ratio": args.minimum_moved_area_ratio,
            "maximum_old_new_overlap": args.maximum_old_new_overlap,
            "causal_effect_mass": args.causal_effect_mass,
            "max_side": args.max_side,
            "resolution_divisor": args.resolution_divisor,
            "hard_event_target_source": (
                "sam_instance_support_with_background_validity"
            ),
            "soft_target_map_source": (
                "edited_gaussian_renderer_footprint"
            ),
        },
        "reviewed_object_count": len(candidates),
        "generated_object_count": len(accepted_objects),
        "rejected_object_count": len(rejected_objects),
        "translation_object_count": translation_object_count,
        "translation_unavailable_object_count": (
            len(accepted_objects) - translation_object_count
        ),
        "view_sample_count": len(sample_records),
        "event_sample_count": event_sample_count,
        "objects": accepted_objects,
        "rejected_objects": rejected_objects,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({
        "reviewed_object_count": len(candidates),
        "generated_object_count": len(accepted_objects),
        "rejected_object_count": len(rejected_objects),
        "translation_object_count": translation_object_count,
        "view_sample_count": len(sample_records),
        "event_sample_count": event_sample_count,
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
