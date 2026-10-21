from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import math
import cv2
import numpy as np
from PIL import Image
import torch
from gsedit_scd.vendor.oscd.gaussian_renderer import render
from gsedit_scd.vendor.oscd.scene.cameras import Camera

def find_image(source, image_name):
    candidates = sorted(
        (source / "reference_scene" / "images").glob(f"{image_name}.*")
    )
    if not candidates:
        raise FileNotFoundError(image_name)
    return candidates[0]


def resized_rgb(path, max_side, resolution_divisor=0):
    image = Image.open(path).convert("RGB")
    if resolution_divisor:
        size = tuple(
            max(1, round(value / resolution_divisor)) for value in image.size
        )
    else:
        scale = min(1.0, max_side / max(image.size))
        size = tuple(max(1, round(value * scale)) for value in image.size)
    if size != image.size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image


def resized_reference_images(source, names, max_side, resolution_divisor=0):
    def load(name):
        return resized_rgb(find_image(source, name), max_side, resolution_divisor)
    with ThreadPoolExecutor(max_workers=4) as reader:
        return dict(zip(names, reader.map(load, names)))


def make_camera(camera_record, image):
    image_tensor = (
        torch.from_numpy(np.asarray(image).copy())
        .permute(2, 0, 1)
        .float()
        .cuda()
        / 255.0
    )
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, :3] = np.asarray(
        camera_record["rotation"], dtype=np.float64
    )
    camera_to_world[:3, 3] = np.asarray(
        camera_record["position"], dtype=np.float64
    )
    world_to_camera = np.linalg.inv(camera_to_world)
    fov_x = 2.0 * math.atan(
        0.5 * float(camera_record["width"]) / float(camera_record["fx"])
    )
    fov_y = 2.0 * math.atan(
        0.5 * float(camera_record["height"]) / float(camera_record["fy"])
    )
    return Camera(
        colmap_id=f"reference_{camera_record['id']}",
        R=world_to_camera[:3, :3].T.astype(np.float32),
        T=world_to_camera[:3, 3].astype(np.float32),
        FoVx=fov_x,
        FoVy=fov_y,
        image=image_tensor,
        gt_alpha_mask=None,
        image_name=camera_record["img_name"],
        uid=f"reference_{camera_record['id']}",
    )


def select_contiguous_views(camera_records, count):
    count = min(count, len(camera_records))
    if count < 2:
        raise ValueError("At least two reference views are required")
    start = max(0, len(camera_records) // 2 - count // 2)
    start = min(start, len(camera_records) - count)
    return list(range(start, start + count))


@torch.no_grad()
def owner_observation(camera, gaussians, pipe, background):
    package = render(
        camera,
        gaussians,
        pipe,
        background,
        return_ray_stats=True,
        return_owner_ids=True,
    )
    ray_stats = package["ray_stats"]
    if ray_stats.ndim != 3 or ray_stats.shape[0] < 9:
        raise RuntimeError("FastGS owner lifting requires 9-channel ray statistics")
    owner_ids = ray_stats[7:9].round().long() - 1
    owner_weights = (
        ray_stats[0:1].clamp(0.0, 1.0)
        * ray_stats[1:3].clamp(0.0, 1.0)
    )
    return (
        owner_ids,
        owner_weights,
        ray_stats[0].clamp(0.0, 1.0),
        package["render"].detach(),
    )


@torch.no_grad()
def lift_mask(mask, owner_ids, owner_weights, num_gaussians):
    flat_mask = mask.to(owner_weights.device).reshape(-1)
    positive_mass = torch.zeros(num_gaussians, device=owner_weights.device)
    total_mass = torch.zeros_like(positive_mass)
    for slot in range(owner_ids.shape[0]):
        owners = owner_ids[slot].reshape(-1)
        weights = owner_weights[slot].reshape(-1)
        valid = (owners >= 0) & (owners < num_gaussians) & (weights > 0.0)
        if not bool(valid.any()):
            continue
        valid_owners = owners[valid]
        valid_weights = weights[valid]
        total_mass.scatter_add_(0, valid_owners, valid_weights)
        positive_mass.scatter_add_(
            0,
            valid_owners,
            valid_weights * flat_mask[valid].float(),
        )
    membership = positive_mass / total_mass.clamp_min(1e-8)
    selected = (positive_mass > 0.0) & (positive_mass >= total_mass - positive_mass)
    return selected, membership, positive_mass, total_mass


@torch.no_grad()
def project_membership(membership, owner_ids, owner_weights):
    numerator = torch.zeros_like(owner_weights[0])
    denominator = torch.zeros_like(numerator)
    for slot in range(owner_ids.shape[0]):
        owners = owner_ids[slot]
        weights = owner_weights[slot]
        valid = (owners >= 0) & (owners < membership.shape[0]) & (weights > 0.0)
        safe_owners = owners.clamp(0, membership.shape[0] - 1)
        numerator += torch.where(valid, weights * membership[safe_owners], 0.0)
        denominator += torch.where(valid, weights, 0.0)
    return torch.where(
        denominator > 0.0,
        numerator / denominator.clamp_min(1e-8),
        torch.zeros_like(numerator),
    )


def overlay(image, mask, color):
    array = np.asarray(image).copy()
    mask = np.asarray(mask, dtype=bool)
    color_array = np.empty_like(array)
    color_array[:] = np.asarray(color, dtype=np.uint8)
    array[mask] = np.round(
        0.45 * array[mask] + 0.55 * color_array[mask]
    ).astype(np.uint8)
    return array


def labeled_panel(image, label, width=504):
    scale = width / image.shape[1]
    resized = cv2.resize(
        image,
        (width, max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    cv2.rectangle(resized, (0, 0), (width, 34), (0, 0, 0), thickness=-1)
    cv2.putText(
        resized,
        label,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return resized

