from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import torch

REPOSITORY_ROOT = Path.cwd()  # repository root: run the tools from the repository root

from gsedit_scd.vendor.oscd.gaussian_renderer import render
from gsedit_scd.vendor.oscd.scene.gaussian_model import GaussianModel
from gsedit_scd.data.mask_geometry import find_image, make_camera, owner_observation, resized_rgb
from gsedit_scd.data.mask_geometry import resized_reference_images
from gsedit_scd.data.membership import aggregate_membership
from gsedit_scd.data.edit_geometry import mask_image, tensor_image, unpack_mask, write_rgb, write_rgb_batch
from gsedit_scd.data.edits import effect_mass_mask, resolve_review, select_object_fragments, write_soft_map
from gsedit_scd.data.lighting_model import canonical_normals, fit_model, nuisance_primitive_colors, reliable_object_gaussians, render_relighting_basis, view_colors





IDENTITY_COEFFICIENT_6 = (1.0, 1.0, 1.0, 0.0, 0.0, 0.0)


IDENTITY_COEFFICIENT_9 = IDENTITY_COEFFICIENT_6 + (0.0, 0.0, 0.0)


TARGET_CONTRACT_CONDITIONS = ("identity", "relighting")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Expand object edits with matched unchanged and relit examples."
        )
    )
    parser.add_argument("--event-root", type=Path, required=True)
    parser.add_argument("--event-list", type=Path, required=True)
    parser.add_argument("--lighting-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lighting-representatives", type=int, default=3)
    parser.add_argument("--lighting-fit-max-side", type=int, default=320)
    parser.add_argument("--causal-effect-mass", type=float, default=0.99)
    parser.add_argument("--maximum-ignore-image-fraction", type=float, default=0.03)
    parser.add_argument("--maximum-ignore-positive-ratio", type=float, default=20.0)
    parser.add_argument("--maximum-outside-effect-fraction", type=float, default=0.20)
    parser.add_argument("--max-side", type=int, default=1008)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def stable_seed(*parts):
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFF


def relative_path(path):
    return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {
        "object_id",
        "object_name",
        "camera_name",
        "event",
        "reference_image",
        "event_image",
        "positive_mask",
        "ignore_mask",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("event list is empty or missing required columns")
    keys = [(row["object_id"], row["camera_name"], row["event"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("event list contains duplicate object/camera/event rows")
    return rows


def load_event_target_contract(row):
    positive = cv2.imread(
        str(REPOSITORY_ROOT / row["positive_mask"]), cv2.IMREAD_GRAYSCALE
    )
    ignore = cv2.imread(
        str(REPOSITORY_ROOT / row["ignore_mask"]), cv2.IMREAD_GRAYSCALE
    )
    if positive is None or ignore is None:
        raise FileNotFoundError("parent positive/ignore mask is missing")
    positive = positive > 127
    ignore = ignore > 127
    if positive.shape != ignore.shape or np.logical_and(positive, ignore).any():
        raise ValueError("invalid parent positive/ignore contract")
    target_score = None
    target_name = row.get("target_map", "")
    if target_name:
        target_score = cv2.imread(
            str(REPOSITORY_ROOT / target_name), cv2.IMREAD_UNCHANGED
        )
        if target_score is None or target_score.ndim != 2:
            raise FileNotFoundError(target_name)
        target_score = target_score.astype(np.float32)
        if float(target_score.max(initial=0.0)) > 1.0:
            target_score /= 65535.0
        target_score = np.clip(target_score, 0.0, 1.0)
        if target_score.shape != positive.shape:
            raise ValueError("parent target/positive shape mismatch")
    return positive, ignore, target_score


def merge_preserved_ignore(positive, source_ignore, effect_support):
    if positive.shape != source_ignore.shape or positive.shape != effect_support.shape:
        raise ValueError("preserved event masks have different shapes")
    if np.logical_and(positive, source_ignore).any():
        raise ValueError("parent positive/ignore overlap")
    halo = np.logical_and(effect_support, ~positive)
    return np.logical_and(np.logical_or(source_ignore, halo), ~positive)


def representative_coefficients(coefficients, count):
    coefficients = np.asarray(coefficients, dtype=np.float32)
    count = min(int(count), coefficients.shape[0])
    if count < 1:
        raise ValueError("at least one lighting representative is required")
    center = np.median(coefficients, axis=0)
    scale = np.median(np.abs(coefficients - center), axis=0)
    scale = np.maximum(scale, 1e-4)
    normalized = (coefficients - center) / scale
    selected = [int(np.argmin(np.square(normalized).sum(axis=1)))]
    while len(selected) < count:
        distance = np.full(coefficients.shape[0], np.inf, dtype=np.float32)
        for index in selected:
            distance = np.minimum(
                distance,
                np.square(normalized - normalized[index]).sum(axis=1),
            )
        distance[selected] = -np.inf
        selected.append(int(np.argmax(distance)))
    return coefficients[selected], selected, center, scale


def choose_alternate_coefficient(
    own_coefficient,
    representatives,
    center,
    scale,
    camera_name,
    seed,
):
    own = (np.asarray(own_coefficient, dtype=np.float32) - center) / scale
    normalized = (representatives - center) / scale
    distance = np.square(normalized - own).sum(axis=1)
    order = np.argsort(-distance)
    rank = stable_seed(seed, camera_name, "lighting") % min(2, len(order))
    return representatives[int(order[rank])]




def relit_colors(colors, normals, coefficient):
    coefficient = torch.as_tensor(
        coefficient, device=colors.device, dtype=colors.dtype
    )
    transformed = nuisance_primitive_colors(colors, normals, coefficient)
    clipped = ((transformed < 0.0) | (transformed > 1.0)).any(dim=1)
    return transformed.clamp(0.0, 1.0).contiguous(), float(clipped.float().mean())




def stored_rgb_effect(first, second):
    difference = np.abs(
        first.astype(np.int16) - second.astype(np.int16)
    ).mean(axis=2)
    return torch.from_numpy(difference.astype(np.float32) / 255.0).cuda()


def effect_audit(effect, positive, ignore):
    envelope = np.logical_or(positive, ignore)
    total = float(effect.sum().item())
    outside_mask = torch.from_numpy(~envelope).to(effect.device)
    outside = float(effect[outside_mask].sum().item())
    positive_pixels = int(positive.sum())
    ignore_pixels = int(ignore.sum())
    image_pixels = int(positive.size)
    return {
        "positive_pixels": positive_pixels,
        "ignore_pixels": ignore_pixels,
        "ignore_image_fraction": ignore_pixels / max(image_pixels, 1),
        "ignore_positive_ratio": ignore_pixels / max(positive_pixels, 1),
        "outside_effect_fraction": outside / max(total, 1e-12),
    }


def accepted_audit(audit, args):
    return (
        audit["positive_pixels"] > 0
        and audit["ignore_image_fraction"] <= args.maximum_ignore_image_fraction
        and audit["ignore_positive_ratio"] <= args.maximum_ignore_positive_ratio
        and audit["outside_effect_fraction"] <= args.maximum_outside_effect_fraction
    )


@torch.no_grad()
def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0.0 < args.causal_effect_mass <= 1.0:
        raise ValueError("--causal-effect-mass must lie in (0, 1]")
    condition_names = (
        TARGET_CONTRACT_CONDITIONS
    )
    rendered_condition_names = tuple(
        condition for condition in condition_names if condition != "identity"
    )
    rows = read_rows(args.event_list)
    event_manifest = json.loads((args.event_root / "manifest.json").read_text())
    lighting_gate = json.loads(args.lighting_gate.read_text())
    source = Path(event_manifest["source"])
    latent_map = torch.load(
        Path(event_manifest["object_map"]), map_location="cpu", weights_only=False
    )
    observation_cache = torch.load(
        Path(event_manifest["observation_cache"]),
        map_location="cpu",
        weights_only=False,
    )
    _, candidate_manifest, candidates = resolve_review(
        Path(event_manifest["semantic_review"])
    )
    candidate_by_id = {
        candidate["object_id"]: candidate for candidate in candidates
    }
    object_manifest_by_id = {
        record["object_id"]: record for record in event_manifest["objects"]
    }
    kept_objects = sorted({row["object_id"] for row in rows})
    missing_candidates = sorted(set(kept_objects) - set(candidate_by_id))
    missing_records = sorted(set(kept_objects) - set(object_manifest_by_id))
    if missing_candidates or missing_records:
        raise ValueError(
            f"missing candidates={missing_candidates}, records={missing_records}"
        )

    camera_records = json.loads(
        (source / "reference_reconstruction" / "cameras.json").read_text()
    )
    record_by_name = {
        Path(str(record["img_name"])).stem: record for record in camera_records
    }
    unique_cameras = sorted({row["camera_name"] for row in rows})
    if set(unique_cameras) - set(record_by_name):
        raise ValueError("event list contains cameras outside the reconstruction")

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
    normals = canonical_normals(
        gaussians, [record["position"] for record in camera_records]
    )
    reliable_objects, reliable_count = reliable_object_gaussians(
        latent_map, gaussian_count, gaussians.get_xyz.device
    )

    args.output.mkdir(parents=True)
    view_root = args.output / "views"
    object_root = args.output / "objects"
    view_root.mkdir()
    object_root.mkdir()
    camera_cache = {}
    reference_images = resized_reference_images(
        source, [record_by_name[name]["img_name"] for name in unique_cameras], args.max_side,
    )

    def get_camera(camera_name):
        if camera_name not in camera_cache:
            record = record_by_name[camera_name]
            image = reference_images.pop(record["img_name"], None)
            if image is None:
                image = resized_rgb(find_image(source, record["img_name"]), args.max_side)
            camera_cache[camera_name] = make_camera(record, image)
        return camera_cache[camera_name]

    directional_admissible = bool(
        lighting_gate.get("reference_gate", {}).get(
            "directional_admissible", False
        )
    )
    coefficient_width = 9 if directional_admissible else 6
    fitted_coefficients = {}
    lighting_fit_records = []
    for index, camera_name in enumerate(unique_cameras):
        camera = get_camera(camera_name)
        base, coverage, basis, object_pixels = render_relighting_basis(
            camera,
            gaussians,
            pipe,
            background,
            normals,
            reliable_objects,
            normal_conditioned=directional_admissible,
        )
        fit_region = (coverage > 0.05) & ~object_pixels
        coefficient, fallback, fit_pixels = fit_model(
            camera.original_image,
            base,
            coverage,
            basis,
            fit_region,
            args.lighting_fit_max_side,
            normal_conditioned=directional_admissible,
        )
        fitted_coefficients[camera_name] = coefficient.detach().cpu().numpy()
        lighting_fit_records.append({
            "camera_name": camera_name,
            "coefficient": coefficient.detach().cpu().tolist(),
            "fallback": fallback,
            "fit_pixels": fit_pixels,
        })
        print(f"[light {index + 1}/{len(unique_cameras)}] {camera_name}", flush=True)
    representatives, representative_indices, coefficient_center, coefficient_scale = (
        representative_coefficients(
            list(fitted_coefficients.values()),
            args.lighting_representatives,
        )
    )

    camera_states = {}
    clipping_records = []
    h0_saturation_records = []
    for index, camera_name in enumerate(unique_cameras):
        nominal = get_camera(camera_name)
        own_coefficient = fitted_coefficients[camera_name]
        source_alternate = choose_alternate_coefficient(
            own_coefficient,
            representatives,
            coefficient_center,
            coefficient_scale,
            camera_name,
            args.seed,
        )
        identity = (
            IDENTITY_COEFFICIENT_9
            if coefficient_width == 9
            else IDENTITY_COEFFICIENT_6
        )
        identity = np.asarray(identity, dtype=np.float32)
        alternate = source_alternate
        state = {
            "identity": {"view": nominal, "coefficient": identity},
            "base_colors": view_colors(gaussians, nominal),
            "alternate_coefficient": alternate.tolist(),
            "source_alternate_coefficient": source_alternate.tolist(),
            "own_fitted_coefficient": own_coefficient.tolist(),
            "h0": {},
        }
        state.update({
            "relighting": {"view": nominal, "coefficient": alternate},
        })
        directory = view_root / camera_name
        directory.mkdir()
        for condition in rendered_condition_names:
            condition_state = state[condition]
            colors, clipped = relit_colors(
                state["base_colors"], normals, condition_state["coefficient"]
            )
            condition_state["colors"] = colors
            condition_state["clipped"] = clipped
            h0 = render(
                condition_state["view"],
                gaussians,
                pipe,
                background,
                override_color=colors,
            )["render"]
            filename = f"{camera_name}_{condition}_h0.png"
            h0_image = tensor_image(h0)
            write_rgb(directory / filename, h0_image)
            saturated = ((h0_image <= 1) | (h0_image >= 254)).any(axis=2)
            h0_saturation_records.append({
                "camera_name": camera_name,
                "condition": condition,
                "fraction": float(saturated.mean()),
            })
            state["h0"][condition] = {
                "path": relative_path(directory / filename),
                "image": h0_image,
            }
            clipping_records.append(clipped)
        camera_states[camera_name] = state
        print(
            f"[view {index + 1}/{len(unique_cameras)}] {camera_name}",
            flush=True,
        )

    components = {
        int(component["component_id"]): component
        for component in latent_map["components"]
    }
    fragments = latent_map["fragments"]
    fragment_view_cache = {}

    def load_fragment_view(fragment):
        view_index = int(fragment["view_index"])
        if view_index not in fragment_view_cache:
            camera_index = latent_map["camera_indices"][view_index]
            record = camera_records[camera_index]
            camera = get_camera(Path(str(record["img_name"])).stem)
            owner_ids, owner_weights, coverage, baseline = owner_observation(
                camera, gaussians, pipe, background
            )
            fragment_view_cache[view_index] = {
                "view_index": view_index,
                "camera_index": camera_index,
                "record": record,
                "camera": camera,
                "owner_ids": owner_ids,
                "owner_weights": owner_weights,
                "coverage": coverage,
                "render": baseline,
            }
        view = dict(fragment_view_cache[view_index])
        view["fragment"] = fragment
        view["sam_mask"] = unpack_mask(
            observation_cache["views"][view_index]["masks"][
                int(fragment["mask_index"])
            ]
        )
        return view

    rows_by_object = {}
    for row in rows:
        rows_by_object.setdefault(row["object_id"], []).append(row)

    output_rows = []
    rejected_rows = []
    nuisance_audits = []
    for object_index, object_id in enumerate(kept_objects):
        candidate = candidate_by_id[object_id]
        object_record = object_manifest_by_id[object_id]
        selected_fragments, _ = select_object_fragments(
            candidate,
            components,
            fragments,
            latent_map.get("hierarchy_candidates", ()),
            candidate_manifest["criteria"],
            3,
        )
        views = [load_fragment_view(fragment) for fragment in selected_fragments]
        anchor = components[int(candidate["anchor_component_id"])]
        object_ids = anchor["gaussian_ids"].long().cuda()
        membership, _ = aggregate_membership(
            views, list(range(len(views))), object_ids, gaussian_count
        )
        ownership = membership[object_ids].clamp(0.0, 1.0)
        motion_ids = object_ids[ownership >= 0.5]
        motion_membership = torch.zeros_like(membership)
        motion_membership[motion_ids] = membership[motion_ids]

        removed_opacity = opacity.clone()
        removed_opacity[object_ids] *= 1.0 - ownership[:, None]
        moved_points = None
        if object_record["translation"] is not None:
            moved_points = points.clone()
            translation = torch.as_tensor(
                object_record["translation"]["final_translation"],
                device=points.device,
                dtype=points.dtype,
            )
            moved_points[motion_ids] += translation

        colour_target = torch.as_tensor(
            object_record["colour"]["target_rgb"],
            device=points.device,
            dtype=points.dtype,
        )
        colour_mix = float(object_record["colour"]["mix"])
        directory = object_root / f"{object_id}_{object_record['name']}"
        directory.mkdir()

        for row in rows_by_object[object_id]:
            if "identity" in condition_names:
                identity_row = dict(row)
                identity_row.update({
                    "condition": "identity",
                    "h0_image": row["reference_image"],
                    "h0_ignore_mask": "",
                })
                output_rows.append(identity_row)
            camera_name = row["camera_name"]
            camera_state = camera_states[camera_name]
            for condition in rendered_condition_names:
                condition_state = camera_state[condition]
                nuisance_view = condition_state["view"]
                coefficient = condition_state["coefficient"]
                colors = camera_state["base_colors"]
                event_colors = colors
                override_opacity = None
                override_xyz = None
                if row["event"] == "removed":
                    override_opacity = removed_opacity
                elif row["event"] == "recoloured":
                    event_colors = colors.clone()
                    event_colors[object_ids] = (
                        colors[object_ids]
                        + colour_mix
                        * ownership[:, None]
                        * (colour_target - colors[object_ids])
                    )
                elif row["event"] == "translated":
                    if moved_points is None:
                        raise RuntimeError(f"{object_id}: translation row without state")
                    override_xyz = moved_points
                else:
                    raise ValueError(f"unknown event: {row['event']}")
                if row["event"] == "recoloured":
                    event_colors, clipped = relit_colors(
                        event_colors, normals, coefficient
                    )
                else:
                    event_colors = condition_state["colors"]
                    clipped = condition_state["clipped"]
                event_image = render(
                    nuisance_view,
                    gaussians,
                    pipe,
                    background,
                    override_color=event_colors,
                    override_opacity=override_opacity,
                    override_xyz=override_xyz,
                )["render"]
                stored_event_image = tensor_image(event_image)
                stored_h0_image = camera_state["h0"][condition]["image"]
                target_score = None
                source_ignore = None
                positive, source_ignore, target_score = (
                    load_event_target_contract(row)
                )
                if target_score is None:
                    raise ValueError(
                        "preserved soft-target contract requires target_map"
                    )
                effect = stored_rgb_effect(
                    stored_event_image, stored_h0_image
                )
                support = effect_mass_mask(effect, args.causal_effect_mass)
                ignore = (
                    merge_preserved_ignore(positive, source_ignore, support)
                )
                audit = effect_audit(effect, positive, ignore)
                audit.update({
                    "object_id": object_id,
                    "camera_name": camera_name,
                    "event": row["event"],
                    "condition": condition,
                    "primitive_colour_clipped_fraction": clipped,
                })
                if not accepted_audit(audit, args):
                    rejected_rows.append(audit)
                    continue

                stem = f"{camera_name}_{row['event']}_{condition}"
                event_path = directory / f"{stem}.png"
                positive_path = directory / f"{stem}_positive_mask.png"
                target_path = (
                    directory / f"{stem}_target_map.png"
                    if target_score is not None else None
                )
                ignore_path = directory / f"{stem}_ignore_mask.png"
                if target_path is not None:
                    write_soft_map(target_path, target_score)
                write_rgb_batch([(event_path, stored_event_image),
                                 (positive_path, mask_image(positive)),
                                 (ignore_path, mask_image(ignore))])
                output_rows.append({
                    "object_id": row["object_id"],
                    "object_name": row["object_name"],
                    "camera_name": camera_name,
                    "event": row["event"],
                    "condition": condition,
                    "reference_image": row["reference_image"],
                    "h0_image": camera_state["h0"][condition]["path"],
                    "event_image": relative_path(event_path),
                    "positive_mask": relative_path(positive_path),
                    "target_map": (
                        relative_path(target_path) if target_path is not None else ""
                    ),
                    "ignore_mask": relative_path(ignore_path),
                    "h0_ignore_mask": "",
                })
                nuisance_audits.append(audit)
                clipping_records.append(clipped)
        print(
            f"[object {object_index + 1}/{len(kept_objects)}] {object_id}",
            flush=True,
        )

    conditions_by_episode = {}
    for row in output_rows:
        key = (row["object_id"], row["camera_name"], row["event"])
        conditions_by_episode.setdefault(key, set()).add(row["condition"])
    complete_episodes = {
        key
        for key, conditions in conditions_by_episode.items()
        if conditions == set(condition_names)
    }
    incomplete_episodes = [
        {
            "object_id": key[0],
            "camera_name": key[1],
            "event": key[2],
            "conditions": sorted(conditions),
        }
        for key, conditions in sorted(conditions_by_episode.items())
        if key not in complete_episodes
    ]
    output_rows = [
        row
        for row in output_rows
        if (row["object_id"], row["camera_name"], row["event"])
        in complete_episodes
    ]

    fieldnames = (
        "object_id",
        "object_name",
        "camera_name",
        "event",
        "condition",
        "reference_image",
        "h0_image",
        "event_image",
        "positive_mask",
        "target_map",
        "ignore_mask",
        "h0_ignore_mask",
    )
    event_list_path = args.output / "training_events.tsv"
    with event_list_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(output_rows)

    condition_counts = {
        condition: sum(row["condition"] == condition for row in output_rows)
        for condition in condition_names
    }
    output_manifest = dict(event_manifest)
    output_manifest.update({
        "parent_event_root": str(args.event_root.resolve()),
        "parent_event_list": str(args.event_list.resolve()),
        "nuisance_dataset": "semantic_object_event_matched_nuisance_v1",
        "conditions": list(condition_names),
        "condition_counts": condition_counts,
        "event_row_count": len(output_rows),
        "rejected_nuisance_row_count": len(rejected_rows),
        "target_gt_used": False,
        "lighting_gate": str(args.lighting_gate.resolve()),
        "lighting_model": (
            "affine_rgb_gain_bias_plus_directional_normal"
            if directional_admissible
            else "affine_rgb_gain_bias"
        ),
        "lighting_fit_camera_count": len(lighting_fit_records),
        "lighting_representative_indices": representative_indices,
        "lighting_representatives": representatives.tolist(),
        "lighting_transport_mode": (
            "alternate_reference_lighting"
        ),
        "settings": {
            **event_manifest["settings"],
            "nuisance_conditions": list(condition_names),
            "lighting_fit_max_side": args.lighting_fit_max_side,
            "lighting_representative_count": len(representatives),
            "causal_effect_mass": args.causal_effect_mass,
            "event_target_source": (
                "preserved_parent_hard_ignore_soft_contract"
            ),
            "preserve_event_target_contract": (
                True
            ),
            "maximum_ignore_image_fraction": args.maximum_ignore_image_fraction,
            "maximum_ignore_positive_ratio": args.maximum_ignore_positive_ratio,
            "maximum_outside_effect_fraction": args.maximum_outside_effect_fraction,
        },
    })
    (args.output / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2)
    )
    audit = {
        "input_rows": len(rows),
        "output_rows": len(output_rows),
        "expected_rows_without_rejection": len(rows) * len(condition_names),
        "condition_counts": condition_counts,
        "rejected_rows": rejected_rows,
        "dropped_incomplete_episodes": incomplete_episodes,
        "nuisance_audits": nuisance_audits,
        "lighting_fits": lighting_fit_records,
        "h0_saturation_records": h0_saturation_records,
        "reliable_object_component_count": reliable_count,
        "mean_primitive_colour_clipped_fraction": float(
            np.mean(clipping_records)
        ),
        "target_gt_used": False,
    }
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({
        "output": str(args.output),
        "event_list": str(event_list_path),
        "condition_counts": condition_counts,
        "rejected_nuisance_rows": len(rejected_rows),
        "mean_primitive_colour_clipped_fraction": audit[
            "mean_primitive_colour_clipped_fraction"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
