"""Fit a complete scene model from reference data, in dependency order."""
import csv
import json
import pickle

import numpy as np
import torch
from transformers import Sam2Model

from gsedit_scd.online.cues import SAM_MODEL
from gsedit_scd.online.evidence import decode_probability
from gsedit_scd.online.geometry import build_gaussian_overlap_graph
from gsedit_scd.online.pipeline import Ablation, OnlineDetector, load_scene
from .collection import collect_split, collect_likelihood, collect_unchanged, collect_prior
from .likelihood import fit_jeffreys_log_lr
from .prior import fit_prior, image_weights, tail_integration_cuts
from .reference import ReferenceObservations
from .samples import collect_suppression_samples, row_key
from .suppression import fit_suppression
from .unchanged import fit_unchanged_responses


def fit_scene(args):
    output = args.output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    with args.events.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    gaussians = load_scene(args.source / "reference_reconstruction/point_cloud/iteration_30000/point_cloud.ply")
    sam = Sam2Model.from_pretrained(SAM_MODEL, local_files_only=True).half().cuda().eval()
    replay = ReferenceObservations(args.source, args.reference_cache, gaussians, sam)
    split = json.loads(args.split.read_text()) if args.split else collect_split(replay, rows)
    (output / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    learning_cameras = split["evidence_learning"]
    rows = [row for row in rows if row["camera_name"] in set(learning_cameras)]
    ablation = Ablation(**({args.without: False} if args.without else {}))

    print("1/5 Fit response suppression", flush=True)
    samples = collect_suppression_samples(replay, rows, learning_cameras)
    suppression = fit_suppression(samples)
    torch.save(suppression, output / "suppression.pt")
    del samples
    suppression = {key: value.cuda() if torch.is_tensor(value) else value for key, value in suppression.items()}

    print("2/5 Fit monotone log likelihood ratio", flush=True)
    # Do not inherit suppression extraction's TF32 setting: the retained
    # likelihood, unchanged-response, and prior replays use FP32 transforms.
    torch.backends.cuda.matmul.allow_tf32 = False
    replay.views.clear()
    replay.measurements = {}
    histograms, collection = collect_likelihood(replay, rows, suppression, ablation.suppression)
    replay.reference_images.clear()
    pooled = {label: sum(value for key, value in histograms.items() if key.endswith("|" + label))
              for label in ("H1", "H0S", "H0B")}
    units = collection["units"]
    fitted = fit_jeffreys_log_lr(pooled["H1"], pooled["H0S"], pooled["H0B"], units["N1"], units["NS"], units["NB"])
    likelihood = {"bin_log_bayes_factor": fitted["table"].tolist(),
                  "fit_coordinate": "post_suppression_response" if ablation.suppression else "raw_response",
                  "source_prior": 0.5}
    (output / "likelihood.json").write_text(json.dumps(likelihood, indent=2) + "\n")
    table = fitted["table"].float().cuda()
    graph, _ = build_gaussian_overlap_graph(gaussians, 8, appearance_aware=True)

    print("3/5 Fit unchanged-response statistics", flush=True)
    observations = collect_unchanged(args.source, split["unchanged_estimation"], gaussians, suppression, table, graph, ablation,
                                     resolution=args.resolution)
    baseline = fit_unchanged_responses(observations["responses"], observations["weights"], observations["surface_graph"])
    torch.save(baseline, output / "unchanged.pt")
    del observations
    detector = OnlineDetector(gaussians, suppression, table, None, None,
                              baseline["summary"]["surface_edge_response_correlation"],
                              0.0, ablation, baseline_mean=baseline["mean"],
                              baseline_variance=baseline["variance"])

    print("4/5 Fit effective pixel prior", flush=True)
    spatial_scores = None
    if args.contour or getattr(args, "contour_proposals", False):
        from .contour import object_split, select_rows
        selected = select_rows(rows, object_split(rows), 20)
        spatial_scores = {row_key(row): None for _, row in selected}
    observations = collect_prior(replay, rows, learning_cameras, detector, spatial_scores=spatial_scores)
    replay.measurements = None
    positive = [value.numpy() for value in observations["positive_scores"]]
    negative = [value.numpy() for value in observations["negative_scores"]]
    cuts = tail_integration_cuts()
    if args.integration_grid:
        with np.load(args.integration_grid) as grid:
            cuts = grid["cuts"]
    weights = image_weights(negative, observations["event_keys"], cuts)
    prior = fit_prior(positive, negative, weights)
    prior["image_weights"] = weights.tolist()
    prior["integration_cuts"] = cuts.tolist()
    prior["ablation"] = args.without
    (output / "prior.json").write_text(json.dumps(prior, indent=2) + "\n")
    image_quality = dict(zip(observations["row_keys"], weights.tolist()))
    del observations, positive, negative
    proposals = None
    if spatial_scores is not None:
        # History updates are prior-independent, so the retained scores remain valid.
        # Apply the newly fitted prior with the same floating decoder as inference.
        # Do not threshold the pre-prior score or use an earlier checkpoint.
        proposals = {
            key: covered & (decode_probability(score, prior["prior_log_odds"]) >= 0.5)
            for key, (score, covered) in spatial_scores.items()
        }
        del spatial_scores
        if getattr(args, "contour_proposals", False):
            torch.save({"prior_log_odds": prior["prior_log_odds"], "masks": proposals},
                       output / "contour_proposals.pt")

    if args.contour:
        print("5/5 Fit contour classifier", flush=True)
        torch.backends.cuda.matmul.allow_tf32 = True
        from .contour import fit_contour
        fitted = fit_contour(args.source, rows, {key: mask.numpy() for key, mask in proposals.items()},
                             image_quality)
        with (output / "contour.pkl").open("wb") as stream:
            pickle.dump(fitted, stream)
    print(f"Scene parameters saved to {output}", flush=True)
