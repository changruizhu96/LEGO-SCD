"""Time PASLCD Full-20 and summarize the existing online benchmark measurements."""
import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys

from evaluate import PASLCD_SCENES


def ensure_decoder(source):
    directory = Path("models/gsedit_scd/sam2_decoder_aot")
    variants = ("box_single", "box_multiple", "point_single", "point_multiple")
    if all(len(list((directory / name).glob("*.so"))) == 1
           and any((directory / name).glob("*.cubin")) for name in variants):
        return
    if directory.exists():
        raise FileExistsError(
            f"Incomplete SAM2 acceleration files in {directory}. "
            "Move this directory aside before retrying; existing files will not be overwritten.")
    images = sorted(path for path in (source / "reference_scene/images").iterdir()
                    if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not images:
        raise FileNotFoundError(f"No reference image for SAM2 compilation in {source}")
    command = [sys.executable, "-m", "gsedit_scd.compile_decoder",
               "--reference-image", str(images[0]), "--output", str(directory)]
    print("Compiling shared SAM2 acceleration files before inference (not training or timed inference).", flush=True)
    print(shlex.join(command), flush=True)
    subprocess.run(command, check=True)


def summarize(jobs, refine):
    runs = []
    for item, repeat, directory in jobs:
        path = directory / "timing.json"
        report = json.loads(path.read_text())
        frames = report["per_frame"]
        if report["frames"] != 25 or len(frames) != 25 or len({row["frame"] for row in frames}) != 25:
            raise ValueError(f"Expected all 25 distinct PASLCD frames: {path}")
        config = report["configuration"]
        if config["refine"] != refine or config["resolution"] != 4 or config["max_frames"] is not None:
            raise ValueError(f"Timing configuration does not match this run: {path}")
        if not math.isfinite(report["sequence_seconds"]) or report["sequence_seconds"] <= 0:
            raise ValueError(f"Invalid sequence time: {path}")
        runs.append(dict(instance=str(item), repeat=repeat, timing_file=str(path),
                         frames=report["frames"], sequence_seconds=report["sequence_seconds"],
                         initialization_seconds=report["initialization_seconds"],
                         edit_initialization_seconds=report["edit_initialization_seconds"],
                         first_frame_ms=report["first_frame_ms"]))
    total_frames = sum(run["frames"] for run in runs)
    total_seconds = sum(run["sequence_seconds"] for run in runs)
    per_instance = {}
    for item in dict.fromkeys(run["instance"] for run in runs):
        selected = [run for run in runs if run["instance"] == item]
        frames = sum(run["frames"] for run in selected)
        seconds = sum(run["sequence_seconds"] for run in selected)
        per_instance[item] = dict(frames=frames, sequence_seconds=seconds, sequence_fps=frames / seconds)
    return dict(sequence_count=len(runs), frames=total_frames, sequence_seconds=total_seconds,
                sequence_fps=total_frames / total_seconds,
                initialization_seconds=sum(run["initialization_seconds"] for run in runs),
                edit_initialization_seconds=sum(run["edit_initialization_seconds"] for run in runs),
                aggregation="Total frames / total sequence seconds; initialization reported separately.",
                per_instance=per_instance, runs=runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("PASLCD",))
    parser.add_argument("--pixel-only", action="store_true")
    parser.add_argument("--scene", choices=PASLCD_SCENES)
    parser.add_argument("--instance", choices=("Instance_1", "Instance_2"))
    parser.add_argument("--repeats", type=int, default=1, help="Independent runs per instance (default: 1).")
    parser.add_argument("--data-root", type=Path, default=Path("data/PASLCD"))
    parser.add_argument("--training-root", type=Path,
                        help="Use <root>/<instance>/<scene>/parameters and edits/training_events.tsv.")
    parser.add_argument("--output", type=Path, default=Path("outputs/timing_full20/PASLCD"))
    parser.add_argument("--summarize-only", action="store_true", help="Summarize existing timing files without inference.")
    args = parser.parse_args()
    if bool(args.scene) != bool(args.instance):
        parser.error("For a single instance, provide both --scene and --instance.")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1.")
    instances = (args.instance,) if args.instance else ("Instance_1", "Instance_2")
    scenes = (args.scene,) if args.scene else PASLCD_SCENES
    mode = "pixel" if args.pixel_only else "full"
    jobs = [(Path(instance) / scene, repeat, args.output / instance / scene / f"{mode}_r{repeat}")
            for repeat in range(1, args.repeats + 1) for instance in instances for scene in scenes]
    if not args.summarize_only:
        for _, _, directory in jobs:
            if directory.exists() and any(directory.iterdir()):
                parser.error(f"Output already exists: {directory}. Use --summarize-only or a new --output.")
        if not args.pixel_only:
            ensure_decoder(args.data_root / jobs[0][0])
        for item, _, directory in jobs:
            source = args.data_root / item
            parameters = (args.training_root / item / "parameters" if args.training_root
                          else Path("pretrained/PASLCD") / item)
            command = [sys.executable, "-m", "gsedit_scd.infer", "--source", str(source),
                       "--reference-ply", str(source / "reference_reconstruction/point_cloud/iteration_30000/point_cloud.ply"),
                       "--parameters", str(parameters), "--benchmark", "--output", str(directory)]
            if args.training_root:
                command.extend(["--warmup-events", str(args.training_root / item / "edits/training_events.tsv")])
            else:
                edits = sorted((Path("examples/initialization/PASLCD") / item).glob("*.png"))
                if not edits:
                    raise FileNotFoundError(f"No bundled initialization edit for {item}")
                command.extend(["--warmup-images", str(edits[0]), str(edits[0])])
            if not args.pixel_only:
                command.append("--refine")
            print(shlex.join(command), flush=True)
            subprocess.run(command, check=True)
    summary = summarize(jobs, refine=not args.pixel_only)
    destination = args.output / (args.instance or "") / (args.scene or "") / f"{mode}_summary.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    for item, result in summary["per_instance"].items():
        print(f"{item}: {result['sequence_fps']:.2f} FPS")
    print(f"{summary['sequence_count']} sequences, {summary['frames']} frames: {summary['sequence_fps']:.2f} FPS")
    print(f"Results: {destination}")


if __name__ == "__main__":
    main()
