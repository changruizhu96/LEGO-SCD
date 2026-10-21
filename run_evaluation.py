"""Run online detection and evaluate PASLCD Full-20 or CCS3D with one command."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

import evaluate as paslcd
import evaluate_ccs3d as ccs3d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("PASLCD", "CCS3D"))
    parser.add_argument("--pixel-only", action="store_true", help="Run and evaluate only the pixel pipeline.")
    parser.add_argument("--scene", help="Evaluate just this scene instead of the complete dataset.")
    parser.add_argument("--instance", choices=("Instance_1", "Instance_2"),
                        help="Required with --scene for PASLCD.")
    parser.add_argument("--data-root", type=Path, help="Default: data/<dataset>.")
    parser.add_argument("--training-root", type=Path,
                        help="Use fitted <root>/<instance>/<scene>/parameters (CCS3D omits instance).")
    parser.add_argument("--output", type=Path,
                        help="Prediction/result root; default: outputs/evaluation/<dataset>/<full or pixel>.")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Score existing <output>/<instance>/<scene>/predictions without inference.")
    args = parser.parse_args()
    names = paslcd.PASLCD_SCENES if args.dataset == "PASLCD" else tuple(ccs3d.CCS3D_FRAME_COUNTS)
    if args.scene and args.scene not in names:
        parser.error(f"Unknown scene {args.scene}; choose from {', '.join(names)}")
    if args.dataset == "PASLCD" and bool(args.scene) != bool(args.instance):
        parser.error("For a single PASLCD instance, provide both --scene and --instance.")
    if args.dataset == "CCS3D" and args.instance:
        parser.error("CCS3D has no --instance option.")

    data = args.data_root or Path("data") / args.dataset
    output = args.output or Path("outputs/evaluation") / args.dataset / ("pixel" if args.pixel_only else "full")
    scenes = (args.scene,) if args.scene else names
    if args.dataset == "PASLCD":
        instances = (args.instance,) if args.instance else ("Instance_1", "Instance_2")
        items = [Path(instance) / scene for instance in instances for scene in scenes]
        stages = ["probability"] if args.pixel_only else ["probability", "refined_probability"]
    else:
        items = [Path(scene) for scene in scenes]
        stages = ["mask"] if args.pixel_only else ["mask", "refined_mask"]

    if not args.evaluate_only:
        # Check every destination before starting; never overwrite a previous run.
        for item in items:
            destination = output / item / "predictions"
            if destination.exists() and any(destination.iterdir()):
                parser.error(f"Predictions already exist: {destination}. Use --evaluate-only or a new --output.")
        for item in items:
            source = data / item
            parameters = (args.training_root / item / "parameters" if args.training_root
                          else Path("pretrained") / args.dataset / item)
            command = [sys.executable, "-m", "gsedit_scd.infer",
                       "--source", str(source), "--resolution", "4" if args.dataset == "PASLCD" else "2",
                       "--reference-ply", str(source / "reference_reconstruction/point_cloud/iteration_30000/point_cloud.ply"),
                       "--parameters", str(parameters), "--output", str(output / item / "predictions")]
            if not args.pixel_only:
                command.append("--refine")
            if args.training_root and args.dataset == "PASLCD":
                command.extend(["--warmup-events", str(args.training_root / item / "edits/training_events.tsv")])
            print(shlex.join(command), flush=True)
            subprocess.run(command, check=True)

    if args.dataset == "PASLCD":
        if args.scene:
            item = items[0]
            result = {stage: paslcd.evaluate(data / item / "gt_mask", output / item / "predictions" / stage)
                      for stage in stages}
            destination = output / item / "metrics.json"
            scores = result
        else:
            result = paslcd.evaluate_full20(data, output, stages)
            destination = output / "metrics.json"
            scores = result["aggregate"]
    else:
        paths = []
        for item in items:
            result = ccs3d.evaluate(data / item, output / item / "predictions", stages)
            destination = output / item / "metrics.json"
            destination.write_text(json.dumps(result, indent=2) + "\n")
            paths.append(destination)
        if not args.scene:
            result = ccs3d.aggregate(paths)
            destination = output / "summary.json"
        scores = {stage: metrics["aggregate"] for stage, metrics in result["stages"].items()}
    destination.write_text(json.dumps(result, indent=2) + "\n")
    for stage, metrics in scores.items():
        iou = metrics["mean_iou"] if args.dataset == "PASLCD" else metrics["iou"]
        f1 = metrics["mean_f1"] if args.dataset == "PASLCD" else metrics["f1"]
        print(f"{stage}: mIoU {100 * iou:.2f}% | F1 {100 * f1:.2f}%")
    print(f"Results: {destination}")


if __name__ == "__main__":
    main()
