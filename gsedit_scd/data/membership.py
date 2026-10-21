from __future__ import annotations
import numpy as np
import torch
from gsedit_scd.vendor.oscd.gaussian_renderer import render
from gsedit_scd.data.mask_geometry import lift_mask

def binary_iou(prediction, target):
    prediction = prediction.bool()
    target = target.to(prediction.device).bool()
    union = torch.logical_or(prediction, target).sum()
    if int(union) == 0:
        return 1.0
    intersection = torch.logical_and(prediction, target).sum()
    return float((intersection.float() / union.float()).item())


def threshold_candidates(score_maps):
    values = torch.cat([score.reshape(-1) for score in score_maps])
    values = values[values > 1e-7]
    if values.numel() == 0:
        return torch.tensor((0.5,), device=score_maps[0].device)
    quantiles = torch.linspace(0.0, 1.0, 257, device=values.device)
    return torch.unique(torch.quantile(values, quantiles)).clamp(1e-7, 1.0)


def fit_threshold(score_maps, targets, training_indices):
    selected_maps = [score_maps[index] for index in training_indices]
    candidates = threshold_candidates(selected_maps)
    best_threshold = float(candidates[0].item())
    best_iou = -1.0
    view_ious = []
    for index in training_indices:
        score = score_maps[index]
        target = targets[index].to(score.device).bool()
        # Right insertion includes score == threshold, matching the >= masks.
        bins = torch.bucketize(score.reshape(-1).contiguous(), candidates, right=True)
        counts = torch.bincount(bins, minlength=candidates.numel() + 1)
        positives = torch.bincount(bins[target.reshape(-1)], minlength=candidates.numel() + 1)
        predicted = counts.flip(0).cumsum(0).flip(0)[1:]
        intersection = positives.flip(0).cumsum(0).flip(0)[1:]
        union = predicted + target.sum() - intersection
        view_ious.append(torch.where(
            union == 0, 1.0,
            intersection.float() / union.clamp_min(1).float(),
        ))
    ious = torch.stack(view_ious, dim=1).cpu().numpy()
    for threshold, per_view in zip(candidates.cpu().numpy(), ious):
        # Match the original NumPy mean of Python floats and tie-breaking order.
        mean_iou = float(np.mean(per_view, dtype=np.float64))
        value = float(threshold)
        if mean_iou > best_iou or (
            abs(mean_iou - best_iou) < 1e-12 and value > best_threshold
        ):
            best_iou = mean_iou
            best_threshold = value
    return best_threshold, best_iou


def aggregate_membership(views, training_indices, object_ids, gaussian_count):
    positive = torch.zeros(gaussian_count, device="cuda")
    total = torch.zeros_like(positive)
    for index in training_indices:
        view = views[index]
        _, _, view_positive, view_total = lift_mask(
            torch.from_numpy(view["sam_mask"]),
            view["owner_ids"],
            view["owner_weights"],
            gaussian_count,
        )
        positive += view_positive
        total += view_total
    membership = torch.zeros_like(positive)
    membership[object_ids] = (
        positive[object_ids] / total[object_ids].clamp_min(1e-8)
    )
    return membership, total[object_ids] > 0.0


@torch.no_grad()
def render_membership(
    camera, gaussians, pipe, background, membership, override_xyz=None
):
    colors = membership[:, None].expand(-1, 3).contiguous()
    return render(
        camera,
        gaussians,
        pipe,
        background,
        override_color=colors,
        override_xyz=override_xyz,
    )["render"].mean(dim=0)


def project_image_pixel(point, camera):
    homogeneous = torch.cat((point, torch.ones(1, device=point.device)))
    clip = homogeneous @ camera.full_proj_transform
    ndc = clip[:2] / clip[3].clamp_min(1e-8)
    return torch.stack((
        0.5 * (ndc[0] + 1.0) * (camera.image_width - 1),
        0.5 * (ndc[1] + 1.0) * (camera.image_height - 1),
    ))


def mask_extent_along(mask, direction):
    rows, columns = np.nonzero(mask)
    if columns.size == 0:
        raise ValueError("Cannot measure translation clearance for an empty mask")
    coordinates = torch.from_numpy(
        np.stack((columns, rows), axis=1)
    ).to(device=direction.device, dtype=direction.dtype)
    projection = coordinates @ direction
    return float((projection.max() - projection.min()).item())


def full_object_clearance_translation(
    initial_translation, centroid, views, gap_extents
):
    def measure(translation):
        diagnostics = []
        for view in views:
            base_pixel = project_image_pixel(centroid, view["camera"])
            moved_pixel = project_image_pixel(
                centroid + translation, view["camera"]
            )
            delta = moved_pixel - base_pixel
            displacement = delta.norm().clamp_min(1e-8)
            extent = mask_extent_along(
                view["sam_mask"], delta / displacement
            )
            diagnostics.append({
                "camera_name": view["record"]["img_name"],
                "projected_object_extent_pixels": extent,
                "required_empty_gap_pixels": gap_extents * extent,
                "required_displacement_pixels": (1.0 + gap_extents) * extent,
                "achieved_displacement_pixels": float(displacement.item()),
                "clearance_pixels": float(displacement.item()) - extent,
            })
        return diagnostics

    translation = initial_translation.clone()
    initial = measure(translation)
    for _ in range(4):
        current = measure(translation)
        scale = max(
            row["required_displacement_pixels"]
            / row["achieved_displacement_pixels"]
            for row in current
        )
        if scale <= 1.0001:
            break
        translation *= scale
    final = measure(translation)
    for initial_row, final_row in zip(initial, final):
        final_row["initial_displacement_pixels"] = initial_row[
            "achieved_displacement_pixels"
        ]
    total_scale = float(
        (translation.norm() / initial_translation.norm().clamp_min(1e-8)).item()
    )
    return translation, total_scale, final

