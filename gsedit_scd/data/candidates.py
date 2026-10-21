from __future__ import annotations
import os
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import cv2
import numpy as np
import torch
from PIL import Image
from gsedit_scd.data.edit_geometry import unpack_mask, write_rgb
from gsedit_scd.data.panels import crop_with_context, fit_panel

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Exhaustively associate stable FastGS/SAM tracks and visualize all "
            "quality-qualified intervention-instance candidates."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--object-map", type=Path, required=True)
    parser.add_argument("--observation-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-diagnostics", action="store_true",
                        help="Save candidate gallery and multiview review images.")
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--tile-width", type=int, default=340)
    parser.add_argument("--tile-height", type=int, default=260)
    parser.add_argument("--review-views-per-candidate", type=int, default=3)
    parser.add_argument("--association-minimum-views", type=int, default=2)
    parser.add_argument("--association-minimum-loo", type=float, default=0.5)
    parser.add_argument("--minimum-depth-auc", type=float, default=0.5)
    parser.add_argument("--minimum-child-containment", type=float, default=0.5)
    parser.add_argument("--minimum-edge-jaccard", type=float, default=0.25)
    parser.add_argument("--maximum-edge-p-value", type=float, default=1e-6)
    parser.add_argument("--candidate-minimum-views", type=int, default=3)
    parser.add_argument("--candidate-minimum-anchor-loo", type=float, default=0.7)
    parser.add_argument("--maximum-cross-candidate-jaccard", type=float, default=0.05)
    return parser.parse_args()


class DisjointSet:
    def __init__(self, values):
        self.parent = {value: value for value in values}

    def find(self, value):
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first, second):
        first = self.find(first)
        second = self.find(second)
        if first != second:
            self.parent[second] = first


def scene_identity(source):
    scene_slug = re.sub(r"[^a-z0-9]+", "_", source.name.lower()).strip("_")
    instance_match = re.fullmatch(
        r"instance[_ -]*(\d+)", source.parent.name, flags=re.IGNORECASE
    )
    if instance_match:
        instance_slug = f"i{instance_match.group(1)}"
        instance_label = f"Instance {instance_match.group(1)}"
    else:
        instance_slug = re.sub(
            r"[^a-z0-9]+", "_", source.parent.name.lower()
        ).strip("_")
        instance_label = source.parent.name.replace("_", " ")
    return f"{scene_slug}_{instance_slug}", f"{source.name} {instance_label}"


def representative_fragments(component, fragments, count):
    candidates = []
    recorded_fragment_ids = set()
    for record in sorted(
        component.get("loo_records") or (),
        key=lambda item: float(item["iou"]),
        reverse=True,
    ):
        fragment_id = int(record["fragment_id"])
        candidates.append((fragments[fragment_id], float(record["iou"])))
        recorded_fragment_ids.add(fragment_id)
    for fragment_id in component["fragment_ids"]:
        fragment_id = int(fragment_id)
        if fragment_id not in recorded_fragment_ids:
            candidates.append((fragments[fragment_id], float("nan")))

    selected = []
    selected_views = set()
    for fragment, iou in candidates:
        view_index = int(fragment["view_index"])
        if view_index in selected_views:
            continue
        selected.append((fragment, iou))
        selected_views.add(view_index)
        if len(selected) == count:
            break
    return selected


def load_image(source, name, shape, cache):
    key = (name, shape)
    if key not in cache:
        paths = sorted((source / "reference_scene" / "images").glob(f"{name}.*"))
        if not paths:
            raise FileNotFoundError(name)
        image = Image.open(paths[0]).convert("RGB")
        if image.size != (shape[1], shape[0]):
            image = image.resize((shape[1], shape[0]), Image.Resampling.LANCZOS)
        cache[key] = np.asarray(image).copy()
    return cache[key]


def make_tile(image, mask, record, width, height):
    crop, crop_mask = crop_with_context(image, mask)
    overlay = crop.astype(np.float32)
    overlay[crop_mask] = 0.45 * overlay[crop_mask] + 0.55 * np.asarray(
        (70, 220, 110), dtype=np.float32
    )
    image_height = height - 72
    tile = fit_panel(overlay.astype(np.uint8), width, image_height)
    label = np.full((72, width, 3), 18, dtype=np.uint8)
    cv2.putText(
        label,
        f"{record['candidate_id']} | anchor {record['anchor_component_id']}",
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (80, 235, 120),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        label,
        (
            f"tracks {len(record['association_component_ids'])} | union views "
            f"{record['associated_view_count']} | anchor views {record['anchor_view_count']}"
        ),
        (8, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        label,
        (
            f"LOO {record['anchor_mean_loo_iou']:.3f} | area "
            f"{100.0 * record['anchor_maximum_area_fraction']:.2f}% | "
            f"cross-J {record['maximum_cross_candidate_jaccard']:.4f}"
        ),
        (8, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )
    return np.concatenate((tile, label), axis=0)


def make_review_panel(image, mask, candidate_id, view_name, iou, width, height):
    crop, crop_mask = crop_with_context(image, mask)
    overlay = crop.astype(np.float32)
    overlay[crop_mask] = 0.45 * overlay[crop_mask] + 0.55 * np.asarray(
        (70, 220, 110), dtype=np.float32
    )
    panel_height = height - 48
    panel = fit_panel(overlay.astype(np.uint8), width, panel_height)
    label = np.full((48, width, 3), 18, dtype=np.uint8)
    cv2.putText(
        label,
        candidate_id,
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (80, 235, 120),
        1,
        cv2.LINE_AA,
    )
    iou_text = "n/a" if not np.isfinite(iou) else f"{iou:.3f}"
    cv2.putText(
        label,
        f"{view_name} | LOO {iou_text}",
        (8, 39),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return np.concatenate((panel, label), axis=0)


def anchor_for_group(member_ids, components, minimum_candidate_views):
    stable = [
        component_id
        for component_id in member_ids
        if int(components[component_id]["view_count"]) >= minimum_candidate_views
    ]
    pool = stable if stable else list(member_ids)
    return max(
        pool,
        key=lambda component_id: (
            float(components[component_id].get("mean_loo_iou") or 0.0),
            int(components[component_id]["view_count"]),
            int(components[component_id]["gaussian_count"]),
        ),
    )


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.columns <= 0 or args.tile_width <= 0 or args.tile_height <= 72:
        raise ValueError("Invalid gallery dimensions")
    if args.review_views_per_candidate <= 0:
        raise ValueError("--review-views-per-candidate must be positive")
    candidate_prefix, scene_label = scene_identity(args.source)
    latent_map = torch.load(args.object_map, map_location="cpu", weights_only=False)
    observation_cache = torch.load(
        args.observation_cache, map_location="cpu", weights_only=False
    )
    if os.path.realpath(latent_map["source"]) != os.path.realpath(str(args.source)):
        raise ValueError("Object-map source does not match --source")
    components = {
        int(component["component_id"]): component
        for component in latent_map["components"]
    }
    fragments = latent_map["fragments"]
    camera_records = json.loads(
        (args.source / "reference_reconstruction" / "cameras.json").read_text()
    )
    expected_names = [
        camera_records[index]["img_name"] for index in latent_map["camera_indices"]
    ]
    if expected_names != [view["name"] for view in observation_cache["views"]]:
        raise ValueError("Object map and observation cache camera names differ")

    association_ids = {
        component_id
        for component_id, component in components.items()
        if int(component["view_count"]) >= args.association_minimum_views
        and float(component.get("mean_loo_iou") or 0.0)
        >= args.association_minimum_loo
        and float(component.get("inverse_depth_boundary_auc") or 0.0)
        > args.minimum_depth_auc
    }
    disjoint_set = DisjointSet(association_ids)
    accepted_edges = []
    for edge in latent_map.get("hierarchy_candidates", ()):
        child = int(edge["child_component"])
        parent = int(edge["parent_component"])
        if child not in association_ids or parent not in association_ids:
            continue
        if (
            float(edge["child_containment"]) >= args.minimum_child_containment
            and float(edge["jaccard"]) >= args.minimum_edge_jaccard
            and float(edge["p_value"]) <= args.maximum_edge_p_value
        ):
            disjoint_set.union(child, parent)
            accepted_edges.append(edge)

    groups = defaultdict(list)
    for component_id in sorted(association_ids):
        groups[disjoint_set.find(component_id)].append(component_id)

    candidates = []
    rejected_groups = []
    for member_ids in groups.values():
        associated_views = sorted({
            int(view_index)
            for component_id in member_ids
            for view_index in components[component_id]["view_indices"]
        })
        anchor_id = anchor_for_group(
            member_ids, components, args.candidate_minimum_views
        )
        anchor = components[anchor_id]
        reasons = []
        if len(associated_views) < args.candidate_minimum_views:
            reasons.append("insufficient_associated_views")
        if (
            float(anchor.get("mean_loo_iou") or 0.0)
            < args.candidate_minimum_anchor_loo
        ):
            reasons.append("low_anchor_loo")
        record = {
            "anchor_component_id": anchor_id,
            "association_component_ids": sorted(member_ids),
            "associated_view_indices": associated_views,
            "associated_view_count": len(associated_views),
            "anchor_view_count": int(anchor["view_count"]),
            "anchor_mean_loo_iou": float(anchor.get("mean_loo_iou") or 0.0),
            "anchor_gaussian_count": int(anchor["gaussian_count"]),
            "anchor_centroid": anchor["centroid"],
            "anchor_maximum_area_fraction": float(
                anchor["maximum_fragment_area_fraction"]
            ),
            "rejection_reasons": reasons,
        }
        (rejected_groups if reasons else candidates).append(record)

    candidates.sort(
        key=lambda record: (
            -record["associated_view_count"],
            -record["anchor_mean_loo_iou"],
            record["anchor_component_id"],
        )
    )
    anchor_ids = [
        np.asarray(
            components[record["anchor_component_id"]]["gaussian_ids"],
            dtype=np.int64,
        )
        for record in candidates
    ]
    pairwise = np.zeros((len(candidates), len(candidates)), dtype=np.float64)
    for first in range(len(candidates)):
        for second in range(first + 1, len(candidates)):
            intersection = np.intersect1d(
                anchor_ids[first], anchor_ids[second], assume_unique=True
            ).size
            union = (
                anchor_ids[first].size + anchor_ids[second].size - intersection
            )
            pairwise[first, second] = pairwise[second, first] = (
                intersection / union if union else 0.0
            )

    overlap_conflicts = []
    for index, record in enumerate(candidates, start=1):
        record["candidate_id"] = f"{candidate_prefix}_candidate_{index:03d}"
        pairwise_index = index - 1
        record["maximum_cross_candidate_jaccard"] = float(
            pairwise[pairwise_index].max()
        )
        record["conflicting_anchor_component_ids"] = [
            candidates[other]["anchor_component_id"]
            for other in range(len(candidates))
            if other != pairwise_index
            and pairwise[pairwise_index, other]
            >= args.maximum_cross_candidate_jaccard
        ]
    for first in range(len(candidates)):
        for second in range(first + 1, len(candidates)):
            if pairwise[first, second] >= args.maximum_cross_candidate_jaccard:
                overlap_conflicts.append({
                    "first_anchor_component_id": candidates[first][
                        "anchor_component_id"
                    ],
                    "second_anchor_component_id": candidates[second][
                        "anchor_component_id"
                    ],
                    "jaccard": float(pairwise[first, second]),
                })

    args.output.mkdir(parents=True)
    review_directory = args.output / "candidate_multiview"
    if args.export_diagnostics:
        review_directory.mkdir()
    image_cache = {}
    tiles = []
    for record in candidates:
        anchor = components[record["anchor_component_id"]]
        selected_fragments = representative_fragments(
            anchor, fragments, args.review_views_per_candidate
        )
        fragment, best_iou = selected_fragments[0]
        view_index = int(fragment["view_index"])
        view = observation_cache["views"][view_index]
        record["representative_view_index"] = view_index
        record["representative_view"] = view["name"]
        record["representative_mask_index"] = int(fragment["mask_index"])
        record["anchor_best_loo_iou"] = best_iou
        if args.export_diagnostics:
            mask = unpack_mask(view["masks"][int(fragment["mask_index"])])
            image = load_image(args.source, view["name"], mask.shape, image_cache)
            tiles.append(make_tile(image, mask, record, args.tile_width, args.tile_height))

        review_panels = []
        review_views = []
        for review_fragment, review_iou in selected_fragments:
            review_view_index = int(review_fragment["view_index"])
            review_view = observation_cache["views"][review_view_index]
            review_mask_index = int(review_fragment["mask_index"])
            review_views.append({
                "view_index": review_view_index,
                "view": review_view["name"],
                "mask_index": review_mask_index,
                "loo_iou": review_iou,
            })
            if args.export_diagnostics:
                review_mask = unpack_mask(review_view["masks"][review_mask_index])
                review_image = load_image(
                    args.source, review_view["name"], review_mask.shape, image_cache
                )
                review_panels.append(make_review_panel(
                    review_image,
                    review_mask,
                    record["candidate_id"],
                    review_view["name"],
                    review_iou,
                    args.tile_width,
                    args.tile_height,
                ))
        record["review_views"] = review_views
        if args.export_diagnostics:
            write_rgb(
                review_directory / f"{record['candidate_id']}.png",
                np.concatenate(review_panels, axis=1),
            )

    if args.export_diagnostics:
        rows = math.ceil(len(tiles) / args.columns)
        canvas = np.full(
            (64 + rows * args.tile_height, args.columns * args.tile_width, 3),
            245,
            dtype=np.uint8,
        )
        cv2.putText(
            canvas,
            (
                f"{scene_label} exhaustive stable-entity candidates | "
                f"{len(candidates)} retained after association and quality checks"
            ),
            (12, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.76,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        for index, tile in enumerate(tiles):
            row, column = divmod(index, args.columns)
            y0 = 64 + row * args.tile_height
            x0 = column * args.tile_width
            canvas[y0 : y0 + args.tile_height, x0 : x0 + args.tile_width] = tile
        write_rgb(args.output / "candidate_gallery.png", canvas)

    manifest = {
        "source": str(args.source),
        "scene_label": scene_label,
        "candidate_prefix": candidate_prefix,
        "object_map": str(args.object_map),
        "observation_cache": str(args.observation_cache),
        "criteria": {
            "association_minimum_views": args.association_minimum_views,
            "association_minimum_loo": args.association_minimum_loo,
            "minimum_depth_auc": args.minimum_depth_auc,
            "minimum_child_containment": args.minimum_child_containment,
            "minimum_edge_jaccard": args.minimum_edge_jaccard,
            "maximum_edge_p_value": args.maximum_edge_p_value,
            "candidate_minimum_views": args.candidate_minimum_views,
            "candidate_minimum_anchor_loo": args.candidate_minimum_anchor_loo,
            "maximum_cross_candidate_jaccard": args.maximum_cross_candidate_jaccard,
        },
        "association_component_count": len(association_ids),
        "accepted_association_edge_count": len(accepted_edges),
        "association_group_count": len(groups),
        "candidate_count": len(candidates),
        "cross_candidate_overlap_conflict_count": len(overlap_conflicts),
        "quality_rejected_count": len(rejected_groups),
        "gallery": "candidate_gallery.png" if args.export_diagnostics else None,
        "candidate_multiview_directory": "candidate_multiview" if args.export_diagnostics else None,
        "candidates": candidates,
        "cross_candidate_overlap_conflicts": overlap_conflicts,
        "quality_rejected": rejected_groups,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({
        "association_component_count": len(association_ids),
        "association_group_count": len(groups),
        "candidate_count": len(candidates),
        "cross_candidate_overlap_conflict_count": len(overlap_conflicts),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
