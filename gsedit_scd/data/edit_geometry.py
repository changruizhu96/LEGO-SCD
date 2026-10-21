from __future__ import annotations
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor
import math
import cv2
import numpy as np
import torch

def component_core_ids(component, fragments):
    repeated = np.concatenate([
        np.asarray(fragments[index]["gaussian_ids"], dtype=np.int64)
        for index in component["fragment_ids"]
    ])
    ids, counts = np.unique(repeated, return_counts=True)
    core = ids[counts * 2 >= component["view_count"]]
    return core if core.size else ids


def evenly_spaced(values, count):
    count = min(count, len(values))
    if count == len(values):
        return values
    indices = np.linspace(0, len(values) - 1, count).round().astype(int)
    return [values[index] for index in np.unique(indices)]


def tensor_image(tensor):
    return (
        tensor.clamp(0.0, 1.0)
        .mul(255.0)
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )


def unpack_mask(record):
    shape = tuple(record["shape"])
    packed = record["packed_mask"].cpu().numpy()
    return np.unpackbits(packed, count=int(np.prod(shape))).reshape(shape).astype(bool)


def mask_image(mask):
    if torch.is_tensor(mask):
        mask = mask.detach().float().cpu().numpy()
    gray = np.round(255.0 * np.clip(mask, 0.0, 1.0)).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def mask_overlay(image, mask, color=(80, 220, 110)):
    if torch.is_tensor(mask):
        mask = mask.detach().float().cpu().numpy()
    image = np.asarray(image).copy().astype(np.float32)
    alpha = 0.58 * np.clip(mask, 0.0, 1.0)[..., None]
    color = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    return np.round((1.0 - alpha) * image + alpha * color).clip(0, 255).astype(np.uint8)


def write_rgb(path, image):
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def write_rgb_batch(images):
    # Bound parallel writes to one view; propagate errors before its metadata is saved.
    with ThreadPoolExecutor(max_workers=4) as writer:
        futures = [writer.submit(write_rgb, path, image) for path, image in images]
        for future in futures:
            future.result()


def project_pixel(point, camera):
    homogeneous = torch.cat((point, torch.ones(1, device=point.device)))
    clip = homogeneous @ camera.full_proj_transform
    ndc = clip[:2] / clip[3].clamp_min(1e-8)
    return torch.stack((
        0.5 * (ndc[0] + 1.0) * (camera.image_width - 1),
        0.5 * (1.0 - ndc[1]) * (camera.image_height - 1),
    ))


def image_plane_translation(
    points, core_ids, camera, area_fraction, magnitude, generator
):
    core = points[core_ids]
    centroid = core.median(dim=0).values
    radii = torch.linalg.vector_norm(core - centroid, dim=1)
    radius = radii.median().clamp_min(1e-4)
    view_direction = centroid - camera.camera_center.to(points)
    view_direction = view_direction / view_direction.norm().clamp_min(1e-8)
    direction = torch.randn(3, generator=generator, device="cpu").to(points)
    direction -= torch.dot(direction, view_direction) * view_direction
    if direction.norm() < 1e-6:
        fallback = torch.tensor([1.0, 0.0, 0.0], device=points.device)
        direction = torch.linalg.cross(view_direction, fallback)
        if direction.norm() < 1e-6:
            fallback = torch.tensor([0.0, 1.0, 0.0], device=points.device)
            direction = torch.linalg.cross(view_direction, fallback)
    direction /= direction.norm().clamp_min(1e-8)
    target_pixels = magnitude * np.sqrt(
        area_fraction * camera.image_width * camera.image_height / np.pi
    )
    base_pixel = project_pixel(centroid, camera)
    pixels_per_world_unit = (
        project_pixel(centroid + direction, camera) - base_pixel
    ).norm().clamp_min(1e-6)
    translation = direction * (target_pixels / pixels_per_world_unit)
    achieved_pixels = (
        project_pixel(centroid + translation, camera) - base_pixel
    ).norm().clamp_min(1e-6)
    translation *= target_pixels / achieved_pixels
    achieved_pixels = (
        project_pixel(centroid + translation, camera) - base_pixel
    ).norm()
    return (
        centroid,
        radius,
        translation,
        float(target_pixels),
        float(achieved_pixels.item()),
    )


def _project_points(points, camera):
    homogeneous = torch.cat((
        points,
        torch.ones((points.shape[0], 1), device=points.device, dtype=points.dtype),
    ), dim=1)
    clip = homogeneous @ camera.full_proj_transform.to(points)
    valid = clip[:, 3] > 1e-8
    safe_w = torch.where(valid, clip[:, 3], torch.ones_like(clip[:, 3]))
    ndc = clip[:, :2] / safe_w[:, None]
    pixels = torch.stack((
        0.5 * (ndc[:, 0] + 1.0) * (camera.image_width - 1),
        0.5 * (1.0 - ndc[:, 1]) * (camera.image_height - 1),
    ), dim=1)
    return pixels, valid & torch.isfinite(pixels).all(dim=1)


def _robust_interval(values):
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        raise ValueError("Cannot measure an empty projected object")
    if values.numel() < 8:
        return values.min(), values.max()
    bounds = torch.quantile(
        values,
        torch.tensor((0.01, 0.99), device=values.device, dtype=values.dtype),
    )
    return bounds[0], bounds[1]


def _interval_overlap_fraction(first, second):
    first_min, first_max = first
    second_min, second_max = second
    intersection = (
        torch.minimum(first_max, second_max)
        - torch.maximum(first_min, second_min)
    ).clamp_min(0.0)
    denominator = torch.minimum(
        (first_max - first_min).clamp_min(1e-6),
        (second_max - second_min).clamp_min(1e-6),
    )
    return intersection / denominator


def multiview_full_clearance_translation(
    points,
    core_ids,
    cameras,
    gap_extents=1.0,
    direction_attempts=16,
    minimum_visible_retention=0.8,
    maximum_old_new_overlap=0.01,
    generator=None,
):
    """Choose a scale-normalized translation observable in every input view."""
    if not cameras:
        raise ValueError("Feasible translation needs at least one camera")
    if gap_extents < 0.0 or direction_attempts < 4:
        raise ValueError("Translation gap and direction count are invalid")
    if not 0.0 <= minimum_visible_retention <= 1.0:
        raise ValueError("Visible-retention threshold must lie in [0, 1]")
    if not 0.0 <= maximum_old_new_overlap < 1.0:
        raise ValueError("Old-new overlap threshold must lie in [0, 1)")
    core = points[core_ids]
    if core.ndim != 2 or core.shape[0] < 2 or core.shape[1] != 3:
        raise ValueError("Feasible translation needs at least two 3-D object points")
    centroid = core.median(dim=0).values
    anchor_center = cameras[0].camera_center.to(points)
    view_direction = centroid - anchor_center
    view_direction /= view_direction.norm().clamp_min(1e-8)
    axes = torch.eye(3, device=points.device, dtype=points.dtype)
    reference_axis = axes[view_direction.abs().argmin()]
    tangent_x = torch.linalg.cross(view_direction, reference_axis, dim=0)
    tangent_x /= tangent_x.norm().clamp_min(1e-8)
    tangent_y = torch.linalg.cross(view_direction, tangent_x, dim=0)
    tangent_y /= tangent_y.norm().clamp_min(1e-8)
    phase = (
        float(torch.rand((), generator=generator).item())
        if generator is not None else 0.0
    )
    directions = []
    for index in range(direction_attempts):
        angle = 2.0 * math.pi * (index + phase) / direction_attempts
        direction = math.cos(angle) * tangent_x + math.sin(angle) * tangent_y
        directions.append(direction / direction.norm().clamp_min(1e-8))

    old_projections = []
    old_centres = []
    old_in_frame = []
    for camera in cameras:
        projected, finite = _project_points(core, camera)
        centre, centre_valid = _project_points(centroid[None], camera)
        inside = (
            finite
            & (projected[:, 0] >= 0.0)
            & (projected[:, 0] <= camera.image_width - 1)
            & (projected[:, 1] >= 0.0)
            & (projected[:, 1] <= camera.image_height - 1)
        )
        old_projections.append(projected)
        old_centres.append(centre[0])
        old_in_frame.append(float(inside.float().mean().item()))
        if not bool(centre_valid[0]) or old_in_frame[-1] <= 0.0:
            raise ValueError("Object is not observable in every translation camera")

    candidates = []
    for direction_index, direction in enumerate(directions):
        world_min, world_max = _robust_interval(core @ direction)
        world_extent = (world_max - world_min).clamp_min(1e-4)
        translation = direction * ((1.0 + gap_extents) * world_extent)
        scalable = True
        for _ in range(6):
            required_scale = 1.0
            for camera, old_pixels, old_centre in zip(
                cameras, old_projections, old_centres
            ):
                moved_centre, valid_centre = _project_points(
                    (centroid + translation)[None], camera
                )
                displacement_vector = moved_centre[0] - old_centre
                displacement = displacement_vector.norm()
                if not bool(valid_centre[0]) or float(displacement.item()) < 1e-5:
                    scalable = False
                    break
                pixel_direction = displacement_vector / displacement
                extent_min, extent_max = _robust_interval(
                    old_pixels @ pixel_direction
                )
                projected_extent = (extent_max - extent_min).clamp_min(1e-4)
                required_scale = max(
                    required_scale,
                    float(
                        ((1.0 + gap_extents) * projected_extent / displacement).item()
                    ),
                )
            if not scalable or required_scale <= 1.0001:
                break
            translation *= required_scale
        if not scalable:
            continue

        view_records = []
        for camera, old_pixels, old_centre, source_visible in zip(
            cameras, old_projections, old_centres, old_in_frame
        ):
            moved_pixels, finite = _project_points(core + translation, camera)
            moved_centre, centre_valid = _project_points(
                (centroid + translation)[None], camera
            )
            displacement_vector = moved_centre[0] - old_centre
            displacement = displacement_vector.norm().clamp_min(1e-8)
            pixel_direction = displacement_vector / displacement
            old_interval = _robust_interval(old_pixels @ pixel_direction)
            moved_interval = _robust_interval(moved_pixels @ pixel_direction)
            old_extent = (old_interval[1] - old_interval[0]).clamp_min(1e-4)
            inside = (
                finite
                & (moved_pixels[:, 0] >= 0.0)
                & (moved_pixels[:, 0] <= camera.image_width - 1)
                & (moved_pixels[:, 1] >= 0.0)
                & (moved_pixels[:, 1] <= camera.image_height - 1)
            )
            moved_visible = float(inside.float().mean().item())
            visible_retention = min(moved_visible / max(source_visible, 1e-6), 1.0)
            x_min, x_max = _robust_interval(moved_pixels[:, 0])
            y_min, y_max = _robust_interval(moved_pixels[:, 1])
            border_margin = min(
                float(x_min.item()) / max(camera.image_width - 1, 1),
                float(camera.image_width - 1 - x_max.item())
                / max(camera.image_width - 1, 1),
                float(y_min.item()) / max(camera.image_height - 1, 1),
                float(camera.image_height - 1 - y_max.item())
                / max(camera.image_height - 1, 1),
            )
            view_records.append({
                "visible_retention": visible_retention,
                "old_new_overlap": float(
                    _interval_overlap_fraction(old_interval, moved_interval).item()
                ),
                "displacement_extents": float((displacement / old_extent).item()),
                "border_margin": border_margin,
                "centre_valid": bool(centre_valid[0]),
            })
        minimum_retention = min(row["visible_retention"] for row in view_records)
        maximum_overlap = max(row["old_new_overlap"] for row in view_records)
        minimum_displacement = min(
            row["displacement_extents"] for row in view_records
        )
        minimum_border = min(row["border_margin"] for row in view_records)
        valid = bool(
            minimum_retention >= minimum_visible_retention
            and maximum_overlap <= maximum_old_new_overlap
            and minimum_displacement >= (1.0 + gap_extents) * 0.995
            and all(row["centre_valid"] for row in view_records)
        )
        candidates.append({
            "direction_index": int(direction_index),
            "translation": translation,
            "translation_world_norm": float(translation.norm().item()),
            "world_displacement_extents": float(
                (translation.norm() / world_extent).item()
            ),
            "minimum_visible_retention": minimum_retention,
            "maximum_old_new_overlap": maximum_overlap,
            "minimum_displacement_extents": minimum_displacement,
            "minimum_border_margin": minimum_border,
            "views": view_records,
            "valid": valid,
        })
    valid_candidates = [candidate for candidate in candidates if candidate["valid"]]
    if not valid_candidates:
        return None, {
            "translation_available": False,
            "candidate_count": len(candidates),
            "valid_candidate_count": 0,
            "gap_extents": float(gap_extents),
            "direction_attempts": int(direction_attempts),
            "small_object_area_weighting": False,
        }
    selected = max(
        valid_candidates,
        key=lambda candidate: (
            candidate["minimum_visible_retention"],
            candidate["minimum_border_margin"],
            -candidate["maximum_old_new_overlap"],
            -candidate["translation_world_norm"],
        ),
    )
    translation = selected.pop("translation")
    return translation, {
        "translation_available": True,
        "candidate_count": len(candidates),
        "valid_candidate_count": len(valid_candidates),
        "gap_extents": float(gap_extents),
        "direction_attempts": int(direction_attempts),
        "small_object_area_weighting": False,
        "selected": selected,
    }


def multiview_observation_aware_translation(
    points,
    core_ids,
    cameras,
    gap_extents=0.5,
    direction_attempts=24,
    minimum_valid_views=2,
    minimum_visible_retention=0.5,
    maximum_old_new_overlap=0.01,
    maximum_seed_views=None,
    generator=None,
):
    """Choose one rigid remove-add move, then retain observable views."""
    if not cameras:
        raise ValueError("Observation-aware translation needs at least one camera")
    if not 1 <= minimum_valid_views <= len(cameras):
        raise ValueError("minimum-valid-views is outside the camera count")
    if maximum_seed_views is not None and maximum_seed_views < minimum_valid_views:
        raise ValueError("maximum-seed-views is below minimum-valid-views")
    core = points[core_ids]
    if core.ndim != 2 or core.shape[0] < 2 or core.shape[1] != 3:
        raise ValueError("Translation needs at least two 3-D object points")
    centroid = core.median(dim=0).values
    required_displacement = 1.0 + gap_extents
    candidates = []
    seed_view_indices = list(range(len(cameras)))
    if maximum_seed_views is not None:
        seed_view_indices = evenly_spaced(
            seed_view_indices, maximum_seed_views
        )

    for subset in combinations(seed_view_indices, minimum_valid_views):
        try:
            translation, subset_report = multiview_full_clearance_translation(
                points,
                core_ids,
                [cameras[index] for index in subset],
                gap_extents=gap_extents,
                direction_attempts=direction_attempts,
                minimum_visible_retention=minimum_visible_retention,
                maximum_old_new_overlap=maximum_old_new_overlap,
                generator=generator,
            )
        except ValueError:
            continue
        if translation is None:
            continue

        view_records = []
        for camera in cameras:
            old_pixels, old_finite = _project_points(core, camera)
            moved_pixels, moved_finite = _project_points(
                core + translation, camera
            )
            old_centre, old_centre_valid = _project_points(
                centroid[None], camera
            )
            moved_centre, moved_centre_valid = _project_points(
                (centroid + translation)[None], camera
            )
            old_inside = (
                old_finite
                & (old_pixels[:, 0] >= 0.0)
                & (old_pixels[:, 0] <= camera.image_width - 1)
                & (old_pixels[:, 1] >= 0.0)
                & (old_pixels[:, 1] <= camera.image_height - 1)
            )
            moved_inside = (
                moved_finite
                & (moved_pixels[:, 0] >= 0.0)
                & (moved_pixels[:, 0] <= camera.image_width - 1)
                & (moved_pixels[:, 1] >= 0.0)
                & (moved_pixels[:, 1] <= camera.image_height - 1)
            )
            source_visible = float(old_inside.float().mean().item())
            moved_visible = float(moved_inside.float().mean().item())
            observable = bool(
                old_centre_valid[0]
                and moved_centre_valid[0]
                and old_finite.sum() >= 2
                and moved_finite.sum() >= 2
                and source_visible > 0.0
            )
            record = {
                "source_observable": observable,
                "source_visible_fraction": source_visible,
                "visible_retention": 0.0,
                "old_new_overlap": 1.0,
                "displacement_extents": 0.0,
                "border_margin": -1.0,
                "valid": False,
            }
            if observable:
                displacement_vector = moved_centre[0] - old_centre[0]
                displacement = displacement_vector.norm()
                if float(displacement.item()) >= 1e-5:
                    pixel_direction = displacement_vector / displacement
                    old_interval = _robust_interval(
                        old_pixels[old_finite] @ pixel_direction
                    )
                    moved_interval = _robust_interval(
                        moved_pixels[moved_finite] @ pixel_direction
                    )
                    old_extent = (
                        old_interval[1] - old_interval[0]
                    ).clamp_min(1e-4)
                    retention = min(
                        moved_visible / max(source_visible, 1e-6), 1.0
                    )
                    x_min, x_max = _robust_interval(
                        moved_pixels[moved_finite, 0]
                    )
                    y_min, y_max = _robust_interval(
                        moved_pixels[moved_finite, 1]
                    )
                    margin = min(
                        float(x_min.item()) / max(camera.image_width - 1, 1),
                        float(camera.image_width - 1 - x_max.item())
                        / max(camera.image_width - 1, 1),
                        float(y_min.item()) / max(camera.image_height - 1, 1),
                        float(camera.image_height - 1 - y_max.item())
                        / max(camera.image_height - 1, 1),
                    )
                    overlap = float(
                        _interval_overlap_fraction(
                            old_interval, moved_interval
                        ).item()
                    )
                    displacement_extents = float(
                        (displacement / old_extent).item()
                    )
                    record.update({
                        "visible_retention": retention,
                        "old_new_overlap": overlap,
                        "displacement_extents": displacement_extents,
                        "border_margin": margin,
                        "valid": bool(
                            moved_visible > 0.0
                            and retention >= minimum_visible_retention
                            and overlap <= maximum_old_new_overlap
                            and displacement_extents
                            >= required_displacement * 0.995
                        ),
                    })
            view_records.append(record)

        valid_indices = [
            index for index, row in enumerate(view_records) if row["valid"]
        ]
        valid_rows = [view_records[index] for index in valid_indices]
        candidates.append({
            "translation": translation,
            "translation_world_norm": float(translation.norm().item()),
            "seed_view_indices": list(subset),
            "valid_view_indices": valid_indices,
            "valid_view_count": len(valid_indices),
            "minimum_valid_retention": min(
                (row["visible_retention"] for row in valid_rows),
                default=0.0,
            ),
            "minimum_valid_border_margin": min(
                (row["border_margin"] for row in valid_rows),
                default=-1.0,
            ),
            "maximum_valid_overlap": max(
                (row["old_new_overlap"] for row in valid_rows),
                default=1.0,
            ),
            "views": view_records,
            "seed_translation": subset_report["selected"],
            "valid": len(valid_indices) >= minimum_valid_views,
        })

    valid_candidates = [
        candidate for candidate in candidates if candidate["valid"]
    ]
    if not valid_candidates:
        return None, {
            "translation_available": False,
            "candidate_count": len(candidates),
            "valid_candidate_count": 0,
            "minimum_valid_views": int(minimum_valid_views),
            "seed_view_indices": seed_view_indices,
            "seed_view_count": len(seed_view_indices),
            "reason": "no_translation_retained_enough_views",
        }
    selected = max(
        valid_candidates,
        key=lambda candidate: (
            candidate["valid_view_count"],
            candidate["minimum_valid_retention"],
            candidate["minimum_valid_border_margin"],
            -candidate["maximum_valid_overlap"],
            -candidate["translation_world_norm"],
        ),
    )
    translation = selected["translation"]
    selected_report = {
        key: value for key, value in selected.items() if key != "translation"
    }
    return translation, {
        "translation_available": True,
        "candidate_count": len(candidates),
        "valid_candidate_count": len(valid_candidates),
        "minimum_valid_views": int(minimum_valid_views),
        "gap_extents": float(gap_extents),
        "direction_attempts": int(direction_attempts),
        "minimum_visible_retention": float(minimum_visible_retention),
        "maximum_old_new_overlap": float(maximum_old_new_overlap),
        "seed_view_indices": seed_view_indices,
        "seed_view_count": len(seed_view_indices),
        "selected": selected_report,
    }

