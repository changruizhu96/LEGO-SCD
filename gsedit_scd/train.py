"""Fit scene parameters from reference-only training observations."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .learning.likelihood import fit_jeffreys_log_lr
from .learning.prior import fit_prior, image_weights, tail_integration_cuts
from .learning.suppression import fit_suppression
from .learning.unchanged import fit_unchanged_responses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    scene = stages.add_parser("scene", help="Generate fitting observations and learn all scene parameters.")
    scene.add_argument("--source", type=Path, required=True)
    scene.add_argument("--events", type=Path, required=True)
    scene.add_argument("--reference-cache", type=Path, required=True)
    scene.add_argument("--resolution", type=int, default=4,
                       help="Reference-image downscale for unchanged-response fitting; match registration (PASLCD: 4, CCS3D: 2).")
    scene.add_argument("--integration-grid", type=Path,
                       help="Optional NPZ with fixed tail-integration cuts for a matched rerun; otherwise generated internally.")
    scene.add_argument("--split", type=Path, help="Optional frozen reference-camera split.")
    scene.add_argument("--contour", action="store_true",
                       help="Fit the contour classifier on source pixel-output proposals with the prior-fit "
                            "image weights and side-symmetric SAM features.")
    scene.add_argument("--contour-proposals", action="store_true",
                       help="Also save the source pixel proposal masks used for contour fitting (contour_proposals.pt).")
    scene.add_argument("--without", choices=("suppression", "mapping", "geometry", "history"))
    scene.add_argument("--output", type=Path, required=True)
    suppression = stages.add_parser("suppression", help="Fit response-only/full logistic models.")
    suppression.add_argument("--samples", type=Path, required=True)
    likelihood = stages.add_parser("likelihood", help="Fit the monotone log likelihood-ratio mapping.")
    likelihood.add_argument("--histograms", type=Path, required=True)
    likelihood.add_argument("--collection", type=Path, required=True)
    unchanged = stages.add_parser("unchanged", help="Fit Beta baselines and spatial coefficient c.")
    unchanged.add_argument("--observations", type=Path, required=True)
    prior = stages.add_parser("prior", help="Fit the final pixel prior with a 0.01 pp source-objective gap.")
    prior.add_argument("--observations", type=Path, required=True,
                       help="Torch file: positive_scores, negative_scores, event_keys, optional integration_cuts; source data only.")
    prior.add_argument("--weights", type=Path,
                       help="Optional retained NPZ file with fixed image weights, for matched ablations.")
    for stage in (suppression, likelihood, unchanged, prior):
        stage.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "scene":
        from .learning.scene import fit_scene
        fit_scene(args)
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.stage == "suppression":
        samples = torch.load(args.samples, map_location="cpu", weights_only=True)
        torch.save(fit_suppression(samples), args.output)
    elif args.stage == "likelihood":
        histograms = torch.load(args.histograms, map_location="cpu", weights_only=True)
        units = json.loads(args.collection.read_text())["units"]
        pooled = {label: sum(value for key, value in histograms.items()
                             if key.startswith("raw|") and key.endswith("|train|" + label))
                  for label in ("H1", "H0S", "H0B")}
        fitted = fit_jeffreys_log_lr(pooled["H1"], pooled["H0S"], pooled["H0B"],
                                   units["N1"], units["NS"], units["NB"])
        args.output.write_text(json.dumps({"bin_log_bayes_factor": fitted["table"].tolist(),
                                          "fit_coordinate": "post_suppression_response",
                                          "source_prior": 0.5}, indent=2) + "\n")
    elif args.stage == "unchanged":
        observations = torch.load(args.observations, map_location="cpu", weights_only=True)
        fitted = fit_unchanged_responses(observations["responses"], observations["weights"],
                                         observations["surface_graph"])
        torch.save(fitted, args.output)
    else:
        observations = torch.load(args.observations, map_location="cpu", weights_only=True)
        positive = [np.sort(np.asarray(value, dtype=np.float32)) for value in observations["positive_scores"]]
        negative = [np.sort(np.asarray(value, dtype=np.float32)) for value in observations["negative_scores"]]
        if args.weights:
            with np.load(args.weights) as saved:
                weights = saved["weights"]
        else:
            cuts = observations.get("integration_cuts", tail_integration_cuts())
            weights = image_weights(negative, observations["event_keys"], cuts)
        result = fit_prior(positive, negative, weights)
        result["image_weights"] = weights.tolist()
        if not args.weights:
            result["integration_cuts"] = np.asarray(cuts).tolist()
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
