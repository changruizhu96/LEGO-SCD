"""Evaluate CCS3D binary predictions against pre/post union masks, or average scene results."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


CCS3D_FRAME_COUNTS = {"livingroom": 5, "desk": 4, "bookcase": 5, "bedroom": 4}


def load_mask(path, size=None):
    with Image.open(path) as source:
        image = source.convert("L")
        if size is not None and image.size != size:
            image = image.resize(size, Image.Resampling.NEAREST)
        return np.asarray(image) > 127


def mean_metrics(rows):
    return {key: float(np.mean([row[key] for row in rows]))
            for key in ("precision", "recall", "f1", "iou")}


def evaluate(source, predictions, stages):
    queries = sorted((source / "inference_scene/images").glob("*.png"))
    if not queries:
        raise FileNotFoundError(f"No query PNGs in {source / 'inference_scene/images'}")
    result = {"scene": source.name, "ground_truth": "pre_union_post", "frame_count": len(queries), "stages": {}}
    for stage in stages:
        rows = []
        for query in queries:
            post = load_mask(source / "gt_mask" / query.name)
            size = (post.shape[1], post.shape[0])
            pre = load_mask(source / "gt_pre_mask" / query.name, size)
            target = pre | post
            predicted = load_mask(predictions / stage / query.name, size)
            tp = int((predicted & target).sum())
            fp = int((predicted & ~target).sum())
            fn = int((~predicted & target).sum())
            rows.append({"frame": query.stem, "tp": tp, "fp": fp, "fn": fn,
                         "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                         "f1": 2 * tp / max(2 * tp + fp + fn, 1), "iou": tp / max(tp + fp + fn, 1)})
        result["stages"][stage] = {"aggregate": mean_metrics(rows), "per_frame": rows}
    return result


def aggregate(paths):
    scenes = [json.loads(path.read_text()) for path in paths]
    if len({scene["scene"] for scene in scenes}) != len(scenes):
        raise ValueError("Provide one result file per scene.")
    names = {scene["scene"] for scene in scenes}
    expected = set(CCS3D_FRAME_COUNTS)
    if names != expected:
        raise ValueError(f"Complete CCS3D requires all four scenes: missing={sorted(expected - names)}, unexpected={sorted(names - expected)}")
    stages = set(scenes[0]["stages"])
    if not stages or not stages <= {"mask", "refined_mask"}:
        raise ValueError("Scene results must contain mask and/or refined_mask predictions.")
    if any(scene["ground_truth"] != "pre_union_post" or set(scene["stages"]) != stages for scene in scenes):
        raise ValueError("Scene results must use the same union-mask protocol and prediction stages.")
    for scene in scenes:
        count = CCS3D_FRAME_COUNTS[scene["scene"]]
        if scene["frame_count"] != count:
            raise ValueError(f"{scene['scene']} requires {count} frames; found {scene['frame_count']}")
        frame_sets = []
        for stage in sorted(stages):
            rows = scene["stages"][stage]["per_frame"]
            frames = {row["frame"] for row in rows}
            if len(rows) != count or len(frames) != count:
                raise ValueError(f"{scene['scene']}/{stage} requires {count} distinct per-frame results")
            frame_sets.append(frames)
        if any(frames != frame_sets[0] for frames in frame_sets[1:]):
            raise ValueError(f"Prediction stages must cover the same frames in {scene['scene']}")
    return {"scenes": [scene["scene"] for scene in scenes], "ground_truth": "pre_union_post",
            "frame_count": sum(scene["frame_count"] for scene in scenes),
            "stages": {stage: {"aggregate": mean_metrics([scene["stages"][stage]["aggregate"] for scene in scenes])}
                       for stage in sorted(stages)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source", type=Path, help="Scene prepared by prepare_ccs3d.py.")
    mode.add_argument("--aggregate", nargs="+", type=Path, metavar="JSON", help="Validate all four scenes (18 frames) and average their result files equally.")
    parser.add_argument("--predictions", type=Path, help="Inference output directory; required with --source.")
    parser.add_argument("--stages", nargs="+", choices=("mask", "refined_mask"), default=["mask", "refined_mask"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.source and not args.predictions:
        parser.error("--source requires --predictions")
    if args.aggregate and args.predictions:
        parser.error("--predictions is only used with --source")
    result = aggregate(args.aggregate) if args.aggregate else evaluate(args.source, args.predictions, args.stages)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for stage, values in result["stages"].items():
        metrics = values["aggregate"]
        print(f"{stage}: mIoU {100 * metrics['iou']:.2f}%, F1 {100 * metrics['f1']:.2f}%")


if __name__ == "__main__":
    main()
