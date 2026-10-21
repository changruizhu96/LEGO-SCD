"""Evaluate saved probability maps with the paper's fixed 0.5 threshold."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


PASLCD_SCENES = (
    "Cantina", "Garden", "Lounge", "Lunch_room", "Meeting_room",
    "Playground", "Porch", "Pots", "Printing_area", "Zen",
)


def image_paths(directory):
    images = {}
    for path in sorted(Path(directory).iterdir()):
        if not path.is_file() or path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        if path.stem in images:
            raise ValueError(f"Duplicate frame name in {directory}: {path.stem}")
        images[path.stem] = path
    if not images:
        raise ValueError(f"No images found in {directory}")
    return images


def require_frames(expected, actual, directory):
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise ValueError(f"Frame mismatch in {directory}: missing={missing}, unexpected={extra}")


def load_probability(path, shape):
    probability = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if probability is None:
        raise FileNotFoundError(path)
    if probability.shape != shape:
        probability = cv2.resize(
            probability, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR
        )
    scale = 65535.0 if probability.dtype == np.uint16 else 255.0
    return probability.astype(np.float64) / scale


def average_precision(target, probability):
    positives = int(target.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-probability, kind="stable")
    sorted_target = target[order]
    true_positives = np.cumsum(sorted_target)
    precision = true_positives / np.arange(1, len(sorted_target) + 1)
    return float(precision[sorted_target == 1].sum() / positives)


def evaluate(ground_truth_dir, probability_dir):
    targets = image_paths(ground_truth_dir)
    predictions = image_paths(probability_dir)
    require_frames(targets, predictions, probability_dir)
    images = []
    for path in targets.values():
        probability_path = predictions[path.stem]
        target_image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if target_image is None:
            raise ValueError(f"Could not read ground-truth image: {path}")
        target = (target_image > 127).astype(np.uint8)
        probability = load_probability(probability_path, target.shape)
        prediction = probability >= 0.5
        tp = int(np.logical_and(prediction, target == 1).sum())
        fp = int(np.logical_and(prediction, target == 0).sum())
        fn = int(np.logical_and(~prediction, target == 1).sum())
        images.append(dict(
            frame=path.stem, tp=tp, fp=fp, fn=fn,
            iou=tp / max(tp + fp + fn, 1),
            f1=2 * tp / max(2 * tp + fp + fn, 1),
            average_precision=average_precision(target.reshape(-1), probability.reshape(-1)),
        ))
    if not images:
        raise RuntimeError("No matching ground-truth and probability images were found")
    return dict(
        images=len(images), threshold=0.5,
        mean_iou=float(np.mean([item["iou"] for item in images])),
        mean_f1=float(np.mean([item["f1"] for item in images])),
        mean_average_precision=float(np.nanmean([item["average_precision"] for item in images])),
        tp=sum(item["tp"] for item in images),
        fp=sum(item["fp"] for item in images),
        fn=sum(item["fn"] for item in images),
        per_image=images,
    )


def evaluate_full20(dataset_root, predictions_root, stages):
    # Check every instance before computing a dataset score.
    jobs = []
    for instance in ("Instance_1", "Instance_2"):
        for scene in PASLCD_SCENES:
            source = dataset_root / instance / scene
            queries = image_paths(source / "inference_scene/images")
            if len(queries) != 25:
                raise ValueError(f"Full-20 requires 25 queries in {source}; found {len(queries)}")
            targets = image_paths(source / "gt_mask")
            require_frames(queries, targets, source / "gt_mask")
            prediction = predictions_root / instance / scene / "predictions"
            for stage in stages:
                require_frames(queries, image_paths(prediction / stage), prediction / stage)
            jobs.append((instance, scene, source, prediction))
    instances = {}
    for instance, scene, source, prediction in jobs:
        instances[f"{instance}/{scene}"] = {
            stage: evaluate(source / "gt_mask", prediction / stage) for stage in stages
        }
    aggregate = {
        stage: {
            metric: float(np.mean([item[stage][metric] for item in instances.values()]))
            for metric in ("mean_iou", "mean_f1")
        }
        for stage in stages
    }
    return dict(instance_count=20, images=500, threshold=0.5,
                aggregation="Equal-weight mean of 20 per-instance means; metrics in [0, 1]",
                aggregate=aggregate, per_instance=instances)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--gt", type=Path, help="Single-instance ground-truth directory.")
    mode.add_argument("--paslcd-root", type=Path, help="Full-20 dataset root containing Instance_1 and Instance_2.")
    parser.add_argument("--probability", type=Path)
    parser.add_argument("--predictions-root", type=Path,
                        help="Full-20 outputs: <root>/<Instance_N>/<scene>/predictions/<stage>.")
    parser.add_argument("--stages", nargs="+", choices=("probability", "refined_probability"),
                        default=["probability", "refined_probability"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.paslcd_root:
        if args.predictions_root is None or args.probability is not None:
            parser.error("Full-20 needs --predictions-root, not --probability.")
        metrics = evaluate_full20(args.paslcd_root, args.predictions_root, args.stages)
    else:
        if args.probability is None or args.predictions_root is not None:
            parser.error("Single-instance evaluation needs --probability, not --predictions-root.")
        metrics = evaluate(args.gt, args.probability)
    result = json.dumps(metrics, indent=2)
    print(result)
    if args.output:
        args.output.write_text(result + "\n")


if __name__ == "__main__":
    main()
