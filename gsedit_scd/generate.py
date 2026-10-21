"""Generate automatic object edits and matched lighting examples from a reference scene."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views", type=int, required=True,
                        help="Number of reference views used for automatic object proposals.")
    parser.add_argument("--observation-cache", type=Path,
                        help="Optional existing SAM2 reference-mask cache.")
    parser.add_argument("--max-side", type=int, default=1008)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--export-diagnostics", action="store_true",
                        help="Also save candidate review images, background maps and edit workflow montages.")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cache = args.observation_cache.resolve() if args.observation_cache else output / "reference_masks.pt"
    latent = output / "objects"
    candidates = output / "candidates"
    selection = output / "selected_objects.json"
    gate = output / "lighting_gate.json"
    edits = output / "edits"
    lighting = output / "lighting"

    def run(stage, *options):
        print(f"[{stage}]", flush=True)
        subprocess.run([sys.executable, "-m", f"gsedit_scd.data.{stage}",
                        *(str(option) for option in options)], check=True)

    run("proposals", "--source", source, "--output", latent,
        "--views", args.views, "--max-side", args.max_side,
        "--points-per-batch", 64, "--sam-model", "facebook/sam2.1-hiera-tiny",
        "--visualizations", 8 if args.export_diagnostics else 0,
        "--observation-cache", cache, "--allow-relocated-observation-cache")
    run("candidates", "--source", source, "--object-map", latent / "latent_object_map.pt",
        "--observation-cache", cache, "--output", candidates,
        *(["--export-diagnostics"] if args.export_diagnostics else []))
    run("selection", "--candidate-manifest", candidates / "manifest.json", "--output", selection)
    run("lighting_gate", "--source", source, "--object-map", latent / "latent_object_map.pt",
        "--output", gate, "--max-side", args.max_side)
    run("edits", "--selected-objects", selection, "--output", edits,
        "--views-per-object", 0, "--translation-gap-extents", 0.5,
        "--translation-direction-attempts", 8, "--maximum-translation-seed-views", 4,
        "--colour-mix", 0.45, "--minimum-semantic-loo-iou", 0.65,
        "--minimum-fold-iou", 0.5, "--minimum-moved-area-ratio", 0,
        "--maximum-old-new-overlap", 0.01, "--causal-effect-mass", 0.99,
        "--minimum-valid-translation-views", 1, "--minimum-translation-visible-retention", 0,
        "--minimum-background-alpha", 0.05, "--minimum-background-provenance", 0.5,
        "--max-side", args.max_side, "--seed", args.seed,
        *(["--export-diagnostics"] if args.export_diagnostics else []))
    run("filter_edits", "--manifest", edits / "manifest.json",
        "--output-list", edits / "training_events.tsv", "--output-audit", edits / "quality.csv",
        "--maximum-ignore-image-fraction", 0.03, "--maximum-ignore-positive-ratio", 20,
        "--boundary-ignore-radius", 5, "--maximum-far-ignore-positive-ratio", 1,
        "--maximum-outside-positive-effect-fraction", 0.20,
        "--maximum-background-invalid-object-fraction", 0.5,
        "--minimum-translation-views-per-object", 1)
    run("lighting", "--event-root", edits, "--event-list", edits / "training_events.tsv",
        "--lighting-gate", gate, "--output", lighting, "--lighting-representatives", 3,
        "--lighting-fit-max-side", 320, "--causal-effect-mass", 0.99,
        "--maximum-ignore-image-fraction", 1, "--maximum-ignore-positive-ratio", 1000000000,
        "--maximum-outside-effect-fraction", 0.20, "--max-side", args.max_side,
        "--seed", args.seed)
    run("filter_pairs", "--input-list", lighting / "training_events.tsv",
        "--output-list", output / "paired_events.tsv", "--output-audit", output / "pair_quality.csv",
        "--boundary-ignore-radius", 5, "--maximum-far-ignore-positive-ratio", 1,
        "--expected-conditions", "identity", "relighting")
    run("filter_candidates", "--input-list", output / "paired_events.tsv",
        "--event-audit", edits / "quality.csv", "--event-manifest", edits / "manifest.json",
        "--output-list", output / "training_events.tsv",
        "--output-summary", output / "candidate_quality.json",
        "--minimum-passing-fraction", 0.5, "--minimum-passing-event-types", 2,
        "--duplicate-containment", 0.9, "--duplicate-jaccard", 0.6)


if __name__ == "__main__":
    main()
