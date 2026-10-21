"""Check references, generate edits, and fit parameters for PASLCD or CCS3D."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

from evaluate import PASLCD_SCENES


CCS3D_VIEWS = {"livingroom": 89, "desk": 96, "bookcase": 94, "bedroom": 160}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("PASLCD", "CCS3D"))
    parser.add_argument("--stage", choices=("all", "reference", "edits", "fit"), default="all",
                        help="Default: run all three stages in order for each scene.")
    parser.add_argument("--scene", help="Process one scene instead of the complete dataset.")
    parser.add_argument("--instance", choices=("Instance_1", "Instance_2"))
    parser.add_argument("--data-root", type=Path, help="Original prepared dataset; default: data/<dataset>.")
    parser.add_argument("--checked-root", type=Path, help="Checked reference scenes; default: data/checked/<dataset>.")
    parser.add_argument("--output", type=Path, help="Training outputs; default: outputs/<dataset>.")
    parser.add_argument("--views", type=int,
                        help="Override proposal views (default: 6 for PASLCD; all reference views for CCS3D).")
    args = parser.parse_args()
    names = PASLCD_SCENES if args.dataset == "PASLCD" else tuple(CCS3D_VIEWS)
    if args.scene and args.scene not in names:
        parser.error(f"Unknown scene {args.scene}; choose from {', '.join(names)}")
    if args.dataset == "PASLCD" and bool(args.scene) != bool(args.instance):
        parser.error("For a single PASLCD instance, provide both --scene and --instance.")
    if args.dataset == "CCS3D" and args.instance:
        parser.error("CCS3D has no --instance option.")
    if args.views is not None and args.views < 2:
        parser.error("--views must be at least 2.")
    data = args.data_root or Path("data") / args.dataset
    checked = args.checked_root or Path("data/checked") / args.dataset
    output = args.output or Path("outputs") / args.dataset
    scenes = (args.scene,) if args.scene else names
    if args.dataset == "PASLCD":
        instances = (args.instance,) if args.instance else ("Instance_1", "Instance_2")
        items = [Path(instance) / scene for instance in instances for scene in scenes]
        resolution, max_side = 4, 1008
    else:
        items = [Path(scene) for scene in scenes]
        resolution, max_side = 2, 800

    stages = ("reference", "edits", "fit") if args.stage == "all" else (args.stage,)
    for item in items:
        source = checked / item
        target = output / item
        views = args.views if args.views is not None else (6 if args.dataset == "PASLCD" else CCS3D_VIEWS[item.name])
        commands = {
            "reference": ["gsedit_scd.register", "--source", data / item, "--reference-images",
                          "--resolution", resolution, "--prepared-source", source,
                          "--output", target / "reference_cues"],
            "edits": ["gsedit_scd.generate", "--source", source, "--views", views,
                      "--max-side", max_side, "--output", target / "edits"],
            "fit": ["gsedit_scd.train", "scene", "--source", source,
                    "--events", target / "edits/training_events.tsv",
                    "--reference-cache", target / "reference_cues", "--resolution", resolution,
                    "--contour", "--output", target / "parameters"],
        }
        for stage in stages:
            command = [sys.executable, "-m", *(str(value) for value in commands[stage])]
            print(f"[{item} / {stage}] {shlex.join(command)}", flush=True)
            subprocess.run(command, check=True)
    print(f"Completed {args.stage}: {len(items)} scene instances. Training outputs: {output}")


if __name__ == "__main__":
    main()
