"""Learn contour correction from automatically edited reference images.

Candidates are the connected components of the source pixel-output decision
(posterior >= 0.5 under the fitted prior) on replayed reference edit sequences;
labels come from the edited positive/ignore masks. Each component's box prompts
SAM2 on the observed and reference images, and the 18 mask/quality features are
made side-symmetric before a weighted logistic fit with Platt calibration.
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch
from transformers import Sam2Model, Sam2Processor

from gsedit_scd.data.mask_geometry import find_image, resized_rgb
from gsedit_scd.online.cues import SAM_MODEL
from gsedit_scd.online.refinement import FEATURE_TRANSFORM, symmetric_side_features
from .samples import row_key

ROOT = Path.cwd()
ALLOW_SHORT_STRATA = True
FEATURE_NAMES = tuple(f"{side}_{group}{rank}" for group in ("logit", "iou", "product")
                      for side in ("strong", "weak") for rank in (1, 2, 3))

def object_split(rows):
    object_ids = sorted({row["object_id"] for row in rows})
    if len(object_ids) < 5:
        raise RuntimeError("continuous contour calibration needs at least five objects")
    return {
        object_id: (
            "validation"
            if index % 5 == 0
            else "calibration"
            if index % 5 == 1
            else "fit"
        )
        for index, object_id in enumerate(object_ids)
    }


def select_rows(rows, split_by_object, count):
    selected = []
    expected = {
        (split, event, condition)
        for split in ("fit", "calibration", "validation")
        for event in ("removed", "recoloured", "translated")
        for condition in ("identity", "relighting")
    }
    for stratum in sorted(expected):
        split, event, condition = stratum
        candidates = sorted(
            (
                row
                for row in rows
                if split_by_object[row["object_id"]] == split
                and row["event"] == event
                and row["condition"] == condition
            ),
            key=lambda row: (
                row["camera_name"], row["object_id"], row["event_image"]
            ),
        )
        if len(candidates) < count:
            if ALLOW_SHORT_STRATA:  # per-item fits: take every available row of a short stratum
                if not candidates:
                    continue
                count_here = len(candidates)
            else:
                raise RuntimeError(f"{stratum} has only {len(candidates)} source rows")
        else:
            count_here = count
        positions = np.linspace(0, len(candidates) - 1, count_here).round().astype(int)
        selected.extend((split, candidates[position]) for position in positions)
    return selected


def load_binary(path, shape):
    image = Image.open(path).convert("L")
    if image.size != (shape[1], shape[0]):
        image = image.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    return np.asarray(image) > 127


def source_observation(source_root, row, real_cache):
    reference_key = ("reference", str(ROOT / row["reference_image"]))
    if reference_key not in real_cache:
        reference = Image.open(ROOT / row["reference_image"]).convert("RGB")
        real_cache[reference_key] = reference, np.asarray(reference, dtype=np.float32) / 255.0
    reference, reference_array = real_cache[reference_key]
    event = Image.open(ROOT / row["event_image"]).convert("RGB")
    if event.size != reference.size:
        raise RuntimeError("source event/reference image shapes do not match")
    camera_name = row["camera_name"]
    if camera_name not in real_cache:
        real = resized_rgb(find_image(source_root, camera_name), 1008)
        if real.size != reference.size:
            real = real.resize(reference.size, Image.Resampling.LANCZOS)
        real_cache[camera_name] = np.asarray(real, dtype=np.float32) / 255.0
    event_array = np.asarray(event, dtype=np.float32) / 255.0
    observed = np.clip(
        real_cache[camera_name] + event_array - reference_array, 0.0, 1.0
    )
    target = Image.fromarray(np.rint(255.0 * observed).astype(np.uint8))
    return target, reference


def exact_components(support):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        support.astype(np.uint8), connectivity=8
    )
    height, width = support.shape
    components = []
    for label in range(1, count):
        x, y, w, h = stats[label, :4]
        x1 = float(x)
        y1 = float(y)
        x2 = float(x + w - 1)
        y2 = float(y + h - 1)
        if x2 <= x1:
            if x2 < width - 1:
                x2 += 1.0
            else:
                x1 = max(0.0, x1 - 1.0)
        if y2 <= y1:
            if y2 < height - 1:
                y2 += 1.0
            else:
                y1 = max(0.0, y1 - 1.0)
        components.append({
            "mask": labels == label,
            "box": [x1, y1, x2, y2],
            "pixels": int(stats[label, cv2.CC_STAT_AREA]),
        })
    return components


@torch.inference_mode()
def encode_images(model, processor, target, reference):
    inputs = processor(images=[target, reference], return_tensors="pt")
    original_sizes = inputs["original_sizes"].clone()
    pixel_values = inputs["pixel_values"].to("cuda", dtype=model.dtype)
    return model.get_image_embeddings(pixel_values), original_sizes


@torch.inference_mode()
def decode_soft_boxes(
    model, processor, image_embeddings, original_sizes, boxes
):
    prompt_inputs = processor(
        original_sizes=original_sizes,
        input_boxes=[boxes, boxes],
        return_tensors="pt",
    )
    prompt_inputs = {
        key: value.to("cuda")
        for key, value in prompt_inputs.items()
        if key == "input_boxes"
    }
    outputs = model(
        image_embeddings=image_embeddings,
        multimask_output=True,
        **prompt_inputs,
    )
    low_resolution = outputs.pred_masks.detach().float()
    masks = processor.post_process_masks(
        low_resolution, original_sizes, binarize=False
    )
    scores = outputs.iou_scores.detach().float().cpu().numpy()
    result_logits = {
        "target": masks[0],
        "reference": masks[1],
    }
    result_scores = {
        "target": scores[0],
        "reference": scores[1],
    }
    return result_logits, result_scores


def component_features(logits, scores, index, support):
    """Side-symmetric SAM features of one component's supported pixels (see FEATURE_NAMES)."""
    ordered_logits = []
    ordered_scores = []
    for side in ("target", "reference"):
        side_scores = np.asarray(scores[side][index], dtype=np.float32)
        order = np.argsort(-side_scores, kind="stable")
        if torch.is_tensor(logits[side]):
            # Transfer only selected pixels; keep feature assembly in NumPy.
            selected = torch.from_numpy(support).to(logits[side].device)
            side_logits = logits[side][index][:, selected].T.cpu().numpy()[:, order]
        else:
            side_logits = np.asarray(
                logits[side][index][order], dtype=np.float32
            )[:, support].T
        side_scores = side_scores[order]
        ordered_logits.append(side_logits)
        ordered_scores.append(np.broadcast_to(
            side_scores[None], side_logits.shape
        ))
    mask_logits = np.concatenate(ordered_logits, axis=1)
    predicted_iou = np.concatenate(ordered_scores, axis=1).astype(
        np.float32, copy=False
    )
    features = np.concatenate((
        mask_logits,
        predicted_iou,
        mask_logits * predicted_iou,
    ), axis=1)
    if features.shape[1] != len(FEATURE_NAMES) or not np.isfinite(features).all():
        raise RuntimeError("invalid continuous SAM feature matrix")
    # Same transform as inference: per group and rank, (max, min) over the two sides.
    return symmetric_side_features(torch.from_numpy(features)).numpy()


def balanced_row_weights(labels):
    weights = np.zeros(labels.shape[0], dtype=np.float64)
    classes = np.unique(labels)
    for label in classes:
        selected = labels == label
        weights[selected] = 1.0 / (len(classes) * int(selected.sum()))
    return weights


def globally_balanced_weights(labels, weights):
    result = weights.astype(np.float64, copy=True)
    for label in (0, 1):
        selected = labels == label
        mass = float(result[selected].sum())
        if mass <= 0.0:
            raise RuntimeError(f"source split has no class {label}")
        result[selected] *= 0.5 / mass
    return result


def fit_models(source, l2_C=None):
    fit_features, fit_labels, fit_weights = source["fit"]
    scaler = StandardScaler()
    scaler.fit(fit_features, sample_weight=fit_weights)
    classifier = LogisticRegression(
        penalty=None, solver="lbfgs", max_iter=500, tol=1e-8
    ) if l2_C is None else LogisticRegression(penalty="l2", C=l2_C, solver="lbfgs", max_iter=500, tol=1e-8)
    classifier.fit(
        scaler.transform(fit_features),
        fit_labels,
        sample_weight=fit_weights,
    )
    calibration_features, calibration_labels, calibration_weights = source[
        "calibration"
    ]
    calibration_score = classifier.decision_function(
        scaler.transform(calibration_features)
    )
    calibrator = LogisticRegression(
        penalty=None, solver="lbfgs", max_iter=500, tol=1e-8
    )
    calibrator.fit(
        calibration_score[:, None],
        calibration_labels,
        sample_weight=calibration_weights,
    )
    return scaler, classifier, calibrator


@torch.inference_mode()
def fit_contour(source, rows, proposals, image_quality):
    """Fit the contour classifier on source pixel-output proposals.

    ``proposals`` maps each selected row key to its source pixel decision mask
    (posterior >= 0.5 under the fitted prior); ``image_quality`` maps every
    learning row key to its prior-fit image weight. Proposals supply only the
    box prompts; the edited positive/ignore masks supply the labels. Per-row
    class balancing is multiplied by the image weight before global balancing.
    """
    model = Sam2Model.from_pretrained(SAM_MODEL, local_files_only=True, dtype=torch.float16).eval().cuda()
    processor = Sam2Processor.from_pretrained(SAM_MODEL, local_files_only=True)
    storage = {split: {"features": [], "labels": [], "weights": []}
               for split in ("fit", "calibration", "validation")}
    selected = select_rows(rows, object_split(rows), 20)
    real_cache = {}
    for index, (split, row) in enumerate(selected):
        key = row_key(row)
        target, reference = source_observation(source, row, real_cache)
        shape = (reference.height, reference.width)
        proposal = np.asarray(proposals[key], dtype=bool)
        if proposal.shape != shape:
            proposal = cv2.resize(proposal.astype(np.uint8), (shape[1], shape[0]),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
        positive = load_binary(ROOT / row["positive_mask"], shape)
        ignore = load_binary(ROOT / row["ignore_mask"], shape)
        components = exact_components(proposal)
        row_features, row_labels = [], []
        if components:
            embeddings, sizes = encode_images(model, processor, target, reference)
            for start in range(0, len(components), 16):
                chunk = components[start:start + 16]
                logits, scores = decode_soft_boxes(model, processor, embeddings, sizes,
                                                   [component["box"] for component in chunk])
                for local, component in enumerate(chunk):
                    valid = component["mask"] & ~ignore
                    if valid.any():
                        row_features.append(component_features(logits, scores, local, valid))
                        row_labels.append(positive[valid].astype(np.int8))
            del embeddings
        if row_features:
            features, labels = np.concatenate(row_features), np.concatenate(row_labels)
            storage[split]["features"].append(features)
            storage[split]["labels"].append(labels)
            storage[split]["weights"].append(balanced_row_weights(labels) * image_quality[key])
        print(f"Contour source observations: {index + 1}/{len(selected)} "
              f"({len(components)} components, {len(row_labels) and sum(map(len, row_labels))} pixels)", flush=True)
    arrays = {}
    for split, values in storage.items():
        labels = np.concatenate(values["labels"])
        arrays[split] = (np.concatenate(values["features"]), labels,
                         globally_balanced_weights(labels, np.concatenate(values["weights"])))
    scaler, classifier, calibrator = fit_models(arrays, l2_C=1.0)
    return {"scaler": scaler, "classifier": classifier, "calibrator": calibrator,
            "feature_transform": FEATURE_TRANSFORM}
