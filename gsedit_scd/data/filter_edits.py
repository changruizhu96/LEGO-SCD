from __future__ import annotations
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import cv2
import numpy as np

REPOSITORY_ROOT = Path.cwd()  # repository root: run the tools from the repository root


EVENT_FIELDS = {
    "removed": (
        "removal_positive_mask",
        "removal_ignore_mask",
        "removal_effect_localization",
        "removal_target_map",
    ),
    "recoloured": (
        "colour_positive_mask",
        "colour_ignore_mask",
        "colour_effect_localization",
        "colour_target_map",
    ),
    "translated": (
        "translation_positive_mask",
        "translation_ignore_mask",
        "translation_effect_localization_in_object_union",
        "translation_target_map",
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Filter semantic-instance event rasters using the retained renderer-"
            "artifact contract and write a reproducible training TSV plus audit CSV."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-list", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    parser.add_argument("--maximum-ignore-image-fraction", type=float, default=0.03)
    parser.add_argument("--maximum-ignore-positive-ratio", type=float, default=20.0)
    parser.add_argument("--boundary-ignore-radius", type=int, default=0)
    parser.add_argument("--maximum-far-ignore-positive-ratio", type=float)
    parser.add_argument("--maximum-outside-positive-effect-fraction", type=float, default=0.20)
    parser.add_argument(
        "--maximum-background-invalid-object-fraction", type=float
    )
    parser.add_argument(
        "--minimum-translation-views-per-object", type=int, default=1
    )
    return parser.parse_args()


def relative_path(path):
    return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()


def load_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask > 127


def split_boundary_ignore(positive, ignore, radius):
    if positive.shape != ignore.shape:
        raise ValueError("Positive and ignore mask shapes differ")
    if radius < 0:
        raise ValueError("Boundary ignore radius must be non-negative")
    if radius == 0:
        near = np.zeros_like(ignore)
        return near, ignore.copy()
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * radius + 1, 2 * radius + 1),
    )
    near_support = cv2.dilate(positive.astype(np.uint8), kernel) > 0
    near = np.logical_and(ignore, near_support)
    far = np.logical_and(ignore, ~near_support)
    return near, far


def enforce_minimum_translation_views(
    training_rows, audit_rows, minimum_views
):
    if minimum_views <= 1:
        return training_rows
    counts = {}
    for row in training_rows:
        if row["event"] == "translated":
            counts[row["object_id"]] = counts.get(row["object_id"], 0) + 1
    sparse = {
        object_id for object_id, count in counts.items()
        if count < minimum_views
    }
    if not sparse:
        return training_rows
    retained = [
        row for row in training_rows
        if not (
            row["event"] == "translated"
            and row["object_id"] in sparse
        )
    ]
    for row in audit_rows:
        if (
            row["status"] == "train"
            and row["event"] == "translated"
            and row["object_id"] in sparse
        ):
            row["status"] = "excluded"
            row["filter_reason"] = "translation_lt_minimum_valid_views"
    return retained


def main():
    args = parse_args()
    if args.output_list.exists() or args.output_audit.exists():
        raise FileExistsError("Output list or audit already exists")
    if args.maximum_ignore_image_fraction < 0.0:
        raise ValueError("--maximum-ignore-image-fraction must be non-negative")
    if args.maximum_ignore_positive_ratio < 0.0:
        raise ValueError("--maximum-ignore-positive-ratio must be non-negative")
    if args.boundary_ignore_radius < 0:
        raise ValueError("--boundary-ignore-radius must be non-negative")
    if (
        args.maximum_far_ignore_positive_ratio is not None
        and args.maximum_far_ignore_positive_ratio < 0.0
    ):
        raise ValueError(
            "--maximum-far-ignore-positive-ratio must be non-negative"
        )
    if not 0.0 <= args.maximum_outside_positive_effect_fraction <= 1.0:
        raise ValueError(
            "--maximum-outside-positive-effect-fraction must lie in [0, 1]"
        )
    if (
        args.maximum_background_invalid_object_fraction is not None
        and not 0.0
        <= args.maximum_background_invalid_object_fraction
        <= 1.0
    ):
        raise ValueError(
            "--maximum-background-invalid-object-fraction must lie in [0, 1]"
        )
    if args.minimum_translation_views_per_object < 1:
        raise ValueError(
            "--minimum-translation-views-per-object must be positive"
        )

    manifest = json.loads(args.manifest.read_text())
    object_root = args.manifest.parent / "objects"
    def inspect_object(object_record):
        training_rows = []
        audit_rows = []
        object_directory = object_root / (
            f"{object_record['object_id']}_{object_record['name']}"
        )
        for sample in object_record["samples"]:
            files = sample["files"]
            for event in sample["available_events"]:
                positive_key, ignore_key, localization_key, target_key = EVENT_FIELDS[event]
                reference_path = object_directory / files["reference"]
                event_path = object_directory / files[event]
                positive_path = object_directory / files[positive_key]
                ignore_path = object_directory / files[ignore_key]
                target_name = files.get(target_key)
                target_path = object_directory / target_name if target_name else None
                for path in (
                    reference_path,
                    event_path,
                    positive_path,
                    ignore_path,
                ):
                    if not path.is_file():
                        raise FileNotFoundError(path)
                if target_path is not None and not target_path.is_file():
                    raise FileNotFoundError(target_path)

                positive = load_mask(positive_path)
                ignore = load_mask(ignore_path)
                if positive.shape != ignore.shape:
                    raise ValueError(f"Mask shape mismatch for {event_path}")
                if np.logical_and(positive, ignore).any():
                    raise ValueError(f"Positive and ignore overlap for {event_path}")
                positive_pixels = int(positive.sum())
                ignore_pixels = int(ignore.sum())
                near_ignore, far_ignore = split_boundary_ignore(
                    positive,
                    ignore,
                    args.boundary_ignore_radius,
                )
                near_ignore_pixels = int(near_ignore.sum())
                far_ignore_pixels = int(far_ignore.sum())
                image_pixels = int(positive.size)
                localization = sample[localization_key]
                if localization is None:
                    raise ValueError(f"Missing {localization_key} for {event_path}")
                outside_effect_fraction = max(0.0, 1.0 - float(localization))
                ignore_image_fraction = ignore_pixels / max(image_pixels, 1)
                ignore_positive_ratio = ignore_pixels / max(positive_pixels, 1)
                far_ignore_positive_ratio = (
                    far_ignore_pixels / max(positive_pixels, 1)
                )

                background_invalid_fraction = float(
                    sample.get("background_invalid_fraction_of_object", 0.0)
                )
                reasons = []
                if positive_pixels <= 0:
                    reasons.append("empty_positive")
                if ignore_image_fraction > args.maximum_ignore_image_fraction:
                    reasons.append("yellow_gt_3pct_image")
                if ignore_positive_ratio > args.maximum_ignore_positive_ratio:
                    reasons.append("yellow_gt_20x_positive")
                if (
                    args.maximum_far_ignore_positive_ratio is not None
                    and far_ignore_positive_ratio
                    > args.maximum_far_ignore_positive_ratio
                ):
                    reasons.append("far_yellow_gt_relative_limit")
                if (
                    outside_effect_fraction
                    > args.maximum_outside_positive_effect_fraction
                ):
                    reasons.append("outside_effect_energy_gt_20pct")
                if (
                    event == "removed"
                    and args.maximum_background_invalid_object_fraction
                    is not None
                    and background_invalid_fraction
                    > args.maximum_background_invalid_object_fraction
                ):
                    reasons.append("background_invalid_gt_majority")

                row = {
                    "object_id": object_record["object_id"],
                    "object_name": object_record["name"],
                    "camera_name": sample["camera_name"],
                    "event": event,
                    "reference_image": relative_path(reference_path),
                    "event_image": relative_path(event_path),
                    "positive_mask": relative_path(positive_path),
                    "target_map": (
                        relative_path(target_path) if target_path is not None else ""
                    ),
                    "ignore_mask": relative_path(ignore_path),
                }
                if not reasons:
                    training_rows.append(row)
                audit_rows.append({
                    "status": "excluded" if reasons else "train",
                    **row,
                    "ignore_pixels": ignore_pixels,
                    "near_ignore_pixels": near_ignore_pixels,
                    "far_ignore_pixels": far_ignore_pixels,
                    "positive_pixels": positive_pixels,
                    "image_pixels": image_pixels,
                    "ignore_fraction_of_image": ignore_image_fraction,
                    "ignore_area_ratio_to_positive": ignore_positive_ratio,
                    "far_ignore_area_ratio_to_positive": far_ignore_positive_ratio,
                    "effect_localization_in_positive": float(localization),
                    "outside_positive_effect_fraction": outside_effect_fraction,
                    "background_invalid_fraction_of_object": (
                        background_invalid_fraction
                    ),
                    "filter_reason": ";".join(reasons),
                })

        return training_rows, audit_rows

    training_rows, audit_rows = [], []
    with ThreadPoolExecutor(max_workers=4) as reader:
        for selected, audited in reader.map(inspect_object, manifest["objects"]):
            training_rows.extend(selected)
            audit_rows.extend(audited)

    training_rows = enforce_minimum_translation_views(
        training_rows,
        audit_rows,
        args.minimum_translation_views_per_object,
    )

    args.output_list.parent.mkdir(parents=True, exist_ok=True)
    args.output_audit.parent.mkdir(parents=True, exist_ok=True)
    training_fields = (
        "object_id",
        "object_name",
        "camera_name",
        "event",
        "reference_image",
        "event_image",
        "positive_mask",
        "target_map",
        "ignore_mask",
    )
    with args.output_list.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=training_fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(training_rows)
    audit_fields = tuple(audit_rows[0])
    with args.output_audit.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_fields)
        writer.writeheader()
        writer.writerows(audit_rows)

    kept_objects = sorted({row["object_id"] for row in training_rows})
    event_counts = {
        event: sum(row["event"] == event for row in training_rows)
        for event in EVENT_FIELDS
    }
    print(json.dumps({
        "input_rows": len(audit_rows),
        "output_rows": len(training_rows),
        "excluded_rows": len(audit_rows) - len(training_rows),
        "kept_object_count": len(kept_objects),
        "event_counts": event_counts,
        "output_list": str(args.output_list),
        "output_audit": str(args.output_audit),
    }, indent=2))


if __name__ == "__main__":
    main()
