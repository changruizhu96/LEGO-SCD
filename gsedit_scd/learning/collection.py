"""Collect fitting observations using only reference images and automatic edits."""
from collections import defaultdict

import torch
from transformers import Sam2Model

from gsedit_scd.data.split import split_reference_cameras
from gsedit_scd.online.cues import SAM_MODEL, frame_cues
from gsedit_scd.online.evidence import SupportField
from gsedit_scd.online.frontend import current_evidence, suppress_response
from gsedit_scd.online.geometry import aggregate_primitive_measurement, render_primitive_measurement_probe
from gsedit_scd.vendor.oscd.dataloaders.image_dataset import ImageDataset
from gsedit_scd.vendor.oscd.scene.cameras import Camera
from gsedit_scd.vendor.oscd.gaussian_renderer import render
from .samples import row_key


def collect_split(replay, rows):
    names = list(replay.records)
    visibility = []
    for name in names:
        view, _ = replay.view(name)
        with torch.enable_grad():
            _, _, support, _ = aggregate_primitive_measurement(
                view, replay.gaussians, replay.pipe, replay.background,
                torch.zeros((view.image_height, view.image_width), device=replay.background.device),
            )
        visibility.append(support >= 1)
    visible = torch.stack(visibility).float()
    common = (visible @ visible.T).double().cpu().numpy()
    event_views = defaultdict(set)
    for row in rows:
        event_views[(row["object_id"], row["event"])].add(names.index(row["camera_name"]))
    larger, smaller = split_reference_cameras(names, common, event_views)
    return {"evidence_learning": larger, "unchanged_estimation": smaller}


def collect_likelihood(replay, rows, suppression, use_suppression=True):
    families = {"appearance_nuisance": lambda row: row["event"] == "recoloured",
                "structural_nuisance": lambda row: row["event"] in ("removed", "translated")}
    histograms = {f"raw|{family}|train|{label}": torch.zeros(65536, dtype=torch.float64)
                  for family in families for label in ("H1", "H0S", "H0B")}
    observable = [row for row in rows if float(replay.mask(row["target_map"], soft=True).sum()) > 0]
    row_weights, control_weights = {}, {}

    def control_key(row):
        return tuple(row[key] for key in ("camera_name", "condition", "reference_image", "h0_image"))

    for family, belongs in families.items():
        selected = [row for row in observable if belongs(row)]
        events = sorted({row["event"] for row in selected})
        for event in events:
            event_rows = [row for row in selected if row["event"] == event]
            conditions = sorted({row["condition"] for row in event_rows})
            for condition in conditions:
                condition_rows = [row for row in event_rows if row["condition"] == condition]
                objects = sorted({row["object_id"] for row in condition_rows})
                for object_id in objects:
                    object_rows = [row for row in condition_rows if row["object_id"] == object_id]
                    weight = 1 / len(events) / len(conditions) / len(objects) / len(object_rows)
                    for row in object_rows:
                        row_weights[(family, row_key(row))] = weight
        controls = {control_key(row): row for row in rows if belongs(row)}
        conditions = sorted({key[1] for key in controls})
        for condition in conditions:
            keys = [key for key in controls if key[1] == condition]
            for key in keys:
                control_weights[(family, key)] = 0.5 / len(conditions) / len(keys)

    fixed_statistics = {}

    def score(row, image_key):
        view = replay.observe(row, image_key)
        name = row["camera_name"]
        if name not in fixed_statistics:
            with torch.enable_grad():
                fixed_statistics[name] = render_primitive_measurement_probe(
                    view, replay.gaussians, replay.pipe, replay.background,
                    return_ray_stats=True, return_owner_ids=True,
                )[2:]
        coverage, rays = fixed_statistics[name]
        response = 0.5 * view.probability_measurement
        if use_suppression:
            response = suppress_response(response, coverage, view.reference_render, rays, suppression)
        return response.squeeze(), coverage.squeeze()

    def add(family, label, values, weights):
        positive = weights > 0
        indices = torch.round(values[positive].float().clamp(0, 1) * 65535).long()
        histograms[f"raw|{family}|train|{label}"] += torch.bincount(
            indices, weights=weights[positive].double(), minlength=65536
        ).cpu()

    controls = {}
    for row in rows:
        controls.setdefault(control_key(row), row)
    if replay.measurements is not None:
        replay.prepare_measurements(observable, list(controls.values()))
    for index, row in enumerate(observable):
        response, coverage = score(row, "event_image")
        valid = (coverage > 0.001) & ~replay.mask(row["ignore_mask"])
        target = replay.mask(row["target_map"], soft=True)[valid].float().clamp(0, 1)
        values, negative = response[valid], target <= 1e-6
        if not bool(negative.any()) or float(target.sum()) <= 0:
            raise ValueError(f"Edit row lacks changed or unchanged observations: {row_key(row)}")
        for family, belongs in families.items():
            if belongs(row):
                weight = row_weights[(family, row_key(row))]
                add(family, "H1", values, target / target.sum() * weight)
                add(family, "H0S", values[negative],
                    torch.full_like(values[negative], 0.5 * weight / int(negative.sum())))
        print(f"Likelihood edit observations: {index + 1}/{len(observable)}", flush=True)
    for key, row in controls.items():
        response, coverage = score(row, "h0_image")
        valid = coverage > 0.001
        if row.get("h0_ignore_mask"):
            valid &= ~replay.mask(row["h0_ignore_mask"])
        values = response[valid]
        for family in families:
            weight = control_weights.get((family, key))
            if weight is not None:
                add(family, "H0B", values, torch.full_like(values, weight / values.numel()))
    count = len({(row["object_id"], row["camera_name"]) for row in observable})
    return histograms, {"units": {"N1": count, "NS": count, "NB": len({row["camera_name"] for row in rows})}}


def collect_unchanged(source, names, gaussians, suppression, likelihood, graph, ablation, resolution=4):
    # Reference baselines use reconstruction poses and the original AREA resize.
    from argparse import Namespace
    pipe = Namespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    background = torch.zeros(3, device=gaussians.get_xyz.device)
    args = Namespace(source_path=str(source), resolution=resolution, test_hold=5,
                     masks_dir="", use_colmap_poses=False, eval_poses=False, num_loader_threads=8)
    dataset = ImageDataset(args, instance="ref")
    sam = Sam2Model.from_pretrained(SAM_MODEL, local_files_only=True).half().cuda()
    sam.get_image_embeddings = torch.compile(sam.get_image_embeddings, mode="max-autotune")
    height, width = dataset.get_image_size()
    dummy = torch.zeros((3, height, width), device="cuda").half()
    for _ in range(10):
        frame_cues(dummy, dummy, sam, height, width)
    indices = {name.rsplit(".", 1)[0]: i for i, name in enumerate(dataset.image_name_list)}
    responses, weights = [], []
    for name in names:
        image, info = dataset[indices[name]]
        pose = info["Rt"].cpu().numpy()
        view = Camera(colmap_id=name, R=pose[:3, :3].T, T=pose[:3, 3],
                      FoVx=info["FovX"], FoVy=info["FovY"], image=image[:3],
                      gt_alpha_mask=None, image_name=name, uid=name)
        with torch.no_grad():
            view.reference_render = render(view, gaussians, pipe, background)["render"]
        view.probability_measurement = frame_cues(image[:3], view.reference_render, sam, height, width)[0]
        current = current_evidence(view, gaussians, pipe, background,
                                   suppression, likelihood,
                                   use_suppression=ablation.suppression,
                                   use_mapping=ablation.mapping, use_geometry=ablation.geometry)
        probability = current.combined
        with torch.enable_grad():
            response, _, support, _ = aggregate_primitive_measurement(
                view, gaussians, pipe, background, probability
            )
        responses.append(response.cpu())
        weights.append((support >= 1).float().cpu())
        print(f"Unchanged-response observation: {name}", flush=True)
    return {"responses": torch.stack(responses), "weights": torch.stack(weights),
            "surface_graph": tuple(value.cpu() for value in graph)}


def collect_prior(replay, rows, names, detector, *, spatial_scores=None):
    groups = defaultdict(list)
    for row in rows:
        groups["__".join(row[key] for key in ("object_id", "event", "condition"))].append(row)
    positive_scores, negative_scores, event_keys, row_keys = [], [], [], []
    for _, sequence in sorted(groups.items()):
        if detector.ablation.history:
            detector.field = SupportField(len(detector.mean), detector.background.device)
        for row in sorted(sequence, key=lambda row: names.index(row["camera_name"])):
            view = replay.observe(row)
            result = detector.predict_then_update(view, source_replay=True)
            if spatial_scores is not None and row_key(row) in spatial_scores:
                spatial_scores[row_key(row)] = (
                    result["score"].detach().cpu(), result["covered"].detach().cpu()
                )
            valid = result["covered"] & ~replay.mask(row["ignore_mask"])
            positive = replay.mask(row["positive_mask"]) & valid
            positive_scores.append(result["score"][positive].sort().values.cpu())
            negative_scores.append(result["score"][valid & ~positive].sort().values.cpu())
            event_keys.append(row["object_id"] + "|" + row["event"])
            row_keys.append(row_key(row))
        print(f"Prior observations: {len(positive_scores)}", flush=True)
    return {"positive_scores": positive_scores, "negative_scores": negative_scores,
            "event_keys": event_keys, "row_keys": row_keys}
