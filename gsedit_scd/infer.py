"""Process query frames causally with fitted scene parameters."""
import argparse
import csv
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
from PIL import Image
import torch

from .online.evidence import encode_evidence, encode_probability
from .online.pipeline import Ablation, OnlineDetector, cached_view, load_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-ply", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--query-cache", type=Path)
    inputs.add_argument("--source", type=Path, help="Prepared scene; register and process each query as it arrives.")
    parser.add_argument("--resolution", type=int, default=4,
                        help="Image downscale when using --source (PASLCD: 4, CCS3D: 2); caches retain their own resolution.")
    parser.add_argument("--parameters", type=Path, required=True,
                        help="Directory produced by gsedit_scd.train scene.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--export-evidence", action="store_true")
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--query-images", type=Path, help="Original query RGB directory for refinement from a cache.")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--warmup-images", type=Path, nargs="+",
                        help="Fixed offline edit RGBs for full-pipeline warmup with --source; resized to the query canvas. Supply at least two observations to exercise history.")
    initialization.add_argument("--warmup-events", type=Path,
                        help="Select the first identity-condition edit from an evidence_learning camera in this training_events.tsv; use it twice for initialization.")
    parser.add_argument("--benchmark", action="store_true",
                        help="Time raw-query inference after initialization, including input waits; save timing.json instead of prediction PNGs. Uses eight CPU threads.")
    args = parser.parse_args()
    if args.warmup_events:
        if not args.source:
            parser.error("--warmup-events requires --source.")
        cameras = set(json.loads((args.parameters / "split.json").read_text())["evidence_learning"])
        with args.warmup_events.open() as stream:
            edit = next((row for row in csv.DictReader(stream, delimiter="\t")
                         if row["condition"] == "identity" and row["camera_name"] in cameras), None)
        if edit is None:
            raise ValueError(f"No identity edit from an evidence-learning camera in {args.warmup_events}")
        args.warmup_images = [Path(edit["event_image"])] * 2
    if args.refine and not (args.source or args.query_images):
        parser.error("Refinement needs --source or --query-images.")
    if args.warmup_images and (not args.source or len(args.warmup_images) < 2):
        parser.error("--warmup-images needs --source and at least two offline edit observations (repeating one is allowed).")
    if args.benchmark and (not args.source or not args.warmup_images):
        parser.error("--benchmark requires --source and --warmup-events or --warmup-images.")
    if args.benchmark and (args.export_evidence or args.query_images):
        parser.error("--benchmark uses raw-query inputs without --export-evidence or --query-images.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output must be empty: {args.output}")
    if args.benchmark:
        torch.set_num_threads(8)
    initialization_started = time.perf_counter()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    # Match the reference-baseline and retained query camera precision.
    torch.backends.cuda.matmul.allow_tf32 = False
    gaussians = load_scene(args.reference_ply)
    device = gaussians.get_xyz.device
    suppression = torch.load(args.parameters / "suppression.pt", map_location=device, weights_only=True)
    likelihood = json.loads((args.parameters / "likelihood.json").read_text())
    table = torch.tensor(likelihood["bin_log_bayes_factor"], dtype=torch.float32, device=device)
    baseline = torch.load(args.parameters / "unchanged.pt", map_location=device, weights_only=True)
    prior = json.loads((args.parameters / "prior.json").read_text())
    without = prior.get("ablation")
    detector = OnlineDetector(gaussians, suppression, table, baseline.get("alpha"), baseline.get("beta"),
                              baseline["summary"]["surface_edge_response_correlation"], prior["prior_log_odds"],
                              Ablation(**({without: False} if without else {})),
                              baseline_mean=baseline.get("mean"), baseline_variance=baseline.get("variance"))
    refiner = None
    if args.refine:
        from .online.refinement import AOT_DECODER_DIRECTORY, ObjectRefiner
        refiner = ObjectRefiner(args.parameters / "contour.pkl", chunk=64,
                                aot_directory=AOT_DECODER_DIRECTORY)
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            # Match the existing raw-query or cached-query execution setting.
            torch.backends.cuda.matmul.allow_tf32 = bool(args.source)
            refiner.warmup()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32

    warmup_stats = {}
    timing = {}
    frame_times = []

    def ready(frame_count):
        torch.cuda.synchronize()
        timing["sequence_started"] = time.perf_counter()
        timing["initialization_seconds"] = timing["sequence_started"] - initialization_started
        timing["expected_frames"] = frame_count

    def warmup(register, height, width):
        field = detector.field
        saved = None if field is None else (field.log_evidence.clone(), field.observation_count.clone(), field.frame_count)
        try:
            for index, path in enumerate(args.warmup_images):
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(path)
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                image = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().to(device) / 255
                info = {"name": path.name, "is_test": False}
                if refiner:
                    with Image.open(path) as source:
                        target = source.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
                    info["refinement_rgb"] = torch.from_numpy(np.asarray(target).copy())
                _, view = register(image, info, index)
                result = detector.predict_then_update(view)
                if refiner:
                    result["refined_probability"] = refiner(result, info["refinement_rgb"].to(device),
                                                           view.reference_render, prior["prior_log_odds"])
                    result["refined_mask"] = result["refined_probability"] >= 0.5
                result["mask"] = result["probability"] >= 0.5
        finally:
            if field is not None:
                # Keep warmed operators, but discard all evidence from edit observations.
                field.log_evidence.copy_(saved[0])
                field.observation_count.copy_(saved[1])
                field.frame_count = saved[2]
        warmup_stats["images"] = [str(path) for path in args.warmup_images]
        return warmup_stats

    def frames():
        if args.source:
            from .online.registration import registered_frames
            yield from registered_frames(str(args.source), resolution=args.resolution,
                                         max_inference_frames=args.max_frames or -1, gaussians=gaussians,
                                         refinement_rgb=bool(refiner) and not args.query_images,
                                         cache_cues=False, warmup=warmup if args.warmup_images else None,
                                         on_ready=ready if args.benchmark else None, quiet=args.benchmark)
        else:
            manifest = json.loads((args.query_cache / "manifest.json").read_text())
            for record in manifest["frames"][:args.max_frames]:
                payload = torch.load(args.query_cache / record["file"], map_location="cpu", weights_only=True)
                view = cached_view(payload, manifest["height"], manifest["width"], gaussians,
                                   detector.pipe, detector.background)
                yield payload, view

    def timed_frames():
        iterator = iter(frames())
        while True:
            started = time.perf_counter()
            try:
                item = next(iterator)
            except StopIteration:
                return
            # The first next() also initializes the frontend; ready() marks its end.
            timing["frame_started"] = timing["sequence_started"] if not frame_times else started
            yield item

    folders = ["probability", "mask"]
    if args.export_evidence:
        folders += ["current", "history", "coverage"]
    if refiner:
        folders += ["refined_probability", "refined_mask"]
    if args.benchmark:
        args.output.mkdir(parents=True, exist_ok=True)
    else:
        for folder in folders:
            (args.output / folder).mkdir(parents=True)
    for payload, view in timed_frames() if args.benchmark else frames():
        result = detector.predict_then_update(view)
        if refiner:
            if args.source and not args.query_images:
                target_uint8 = view.refinement_rgb.to(device)
            else:
                directory = args.query_images or args.source / "inference_scene/images"
                matches = sorted(directory.glob(payload["image_name"] + ".*"))
                if not matches:
                    raise FileNotFoundError(directory / payload["image_name"])
                target = Image.open(matches[0]).convert("RGB").resize(
                    (view.image_width, view.image_height), Image.Resampling.LANCZOS
                )
                target_uint8 = torch.from_numpy(np.asarray(target).copy()).to(device)
            result["refined_probability"] = refiner(result, target_uint8, view.reference_render, prior["prior_log_odds"])
        result["mask"] = result["probability"] >= 0.5
        if refiner:
            result["refined_mask"] = result["refined_probability"] >= 0.5
        if args.benchmark:
            torch.cuda.synchronize()
            completed = time.perf_counter()
            frame_times.append(dict(frame=payload["image_name"],
                                    latency_ms=1000 * (completed - timing["frame_started"]),
                                    sequence_elapsed_seconds=completed - timing["sequence_started"]))
            continue
        for folder in folders:
            path = args.output / folder / (payload["image_name"] + ".png")
            value = result[folder]
            if folder.endswith("mask"):
                image = value.cpu().numpy().astype(np.uint8) * 255
            elif folder in ("current", "history", "coverage"):
                image = encode_evidence(value)
            else:
                image = encode_probability(value)
            if not cv2.imwrite(str(path), image):
                raise OSError(f"Could not write {path}")
        print(payload["image_name"], flush=True)
    if args.benchmark:
        if not frame_times or len(frame_times) != timing["expected_frames"]:
            raise RuntimeError("Incomplete timed sequence; no timing summary written")
        elapsed = frame_times[-1]["sequence_elapsed_seconds"]
        report = dict(
            frames=len(frame_times), initialization_seconds=timing["initialization_seconds"],
            edit_initialization_seconds=warmup_stats["seconds"],
            sequence_seconds=elapsed, sequence_fps=len(frame_times) / elapsed,
            first_frame_ms=frame_times[0]["latency_ms"],
            mean_frame_ms=float(np.mean([item["latency_ms"] for item in frame_times])),
            per_frame=frame_times,
            protocol="After initialization to final mask ready, all frames included; includes input waits and inter-frame overhead; excludes PNG export and evaluation. Native prefetch starts during initialization. CUDA synchronized at each mask output.",
            configuration=dict(source=str(args.source), reference_ply=str(args.reference_ply),
                               parameters=str(args.parameters), resolution=args.resolution,
                               refine=args.refine, max_frames=args.max_frames, seed=0,
                               cpu_threads=torch.get_num_threads(), gpu=torch.cuda.get_device_name(device),
                               torch_version=torch.__version__, cuda_version=torch.version.cuda,
                               warmup_images=[str(path) for path in args.warmup_images]),
        )
        (args.output / "timing.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{len(frame_times)} frames: {report['sequence_fps']:.2f} FPS; initialization {report['initialization_seconds']:.3f} s", flush=True)
    if warmup_stats:
        (args.output / "warmup.json").write_text(json.dumps(warmup_stats, indent=2) + "\n")
        print(f"Full-pipeline edit warmup: {warmup_stats['seconds']:.3f} s", flush=True)


if __name__ == "__main__":
    main()
